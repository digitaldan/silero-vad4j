/*
 * #%L
 * silero-vad4j
 * %%
 * Copyright (C) 2026 Contributors to silero-vad4j
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */
package io.github.jvoiceproject.silerovad;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Objects;

/**
 * Silero VAD
 *
 * <p>Plain Java inference of the <a href="https://github.com/snakers4/silero-vad">Silero VAD</a>
 * model for 16 kHz mono audio. Returns the speech probability of each {@link #FRAME_SIZE} sample
 * frame. The weights are exported from the official v6.2.3 ONNX model, which uses the v5
 * architecture.
 *
 * <p>The model keeps state between frames, so an instance should be dedicated to a single
 * continuous audio stream. Call {@link #reset()} before reusing it for a new stream. All methods
 * are safe to be called concurrently. Weights are loaded once and shared by all instances.
 *
 * @author Dan Cunningham - Initial contribution
 */
public class SileroVAD {
    /** Supported sample rate in Hz. */
    public static final int SAMPLE_RATE = 16000;

    /** Number of samples per frame, 32 ms at {@link #SAMPLE_RATE}. */
    public static final int FRAME_SIZE = 512;

    /** Number of samples preserved from the previous frame as preceding context. */
    private static final int CONTEXT_SIZE = 64;

    /** Number of samples padded onto the end of the input buffer using reflect padding. */
    private static final int PAD = 64;

    /** Window size for the Short-Time Fourier Transform (STFT). */
    private static final int WINDOW = 256;

    /** Hop length (step size) between consecutive STFT analysis frames. */
    private static final int HOP = 128;

    /** Number of frequency bins produced by the STFT ({@code WINDOW / 2 + 1}). */
    private static final int BINS = 129;

    /** Hidden state & LSTM input size used in the LSTM and dense output projection layers. */
    private static final int HIDDEN = 128;

    private static final String WEIGHTS = "silero_vad_16k.bin";

    private static Weights sharedWeights;

    private final Weights w;

    /** Internal input buffer holding both preceding context and current frame samples. */
    private final float[] input = new float[CONTEXT_SIZE + FRAME_SIZE];

    /**
     * Padded input buffer containing preceding context, current frame samples, and right-side
     * reflect padding.
     */
    private final float[] padded = new float[CONTEXT_SIZE + FRAME_SIZE + PAD];

    /** Hidden state vector {@code h} of the recurrent LSTM cell. */
    private final float[] h = new float[HIDDEN];

    /** Cell state vector {@code c} of the recurrent LSTM cell. */
    private final float[] c = new float[HIDDEN];

    /**
     * Working buffer for the concatenated LSTM gate pre-activations (input, forget, cell, output).
     */
    private final float[] gates = new float[4 * HIDDEN];

    /**
     * Creates a new detector and initializes (or reuses) the shared model weights.
     *
     * @throws IOException when the bundled weights can't be loaded
     */
    public SileroVAD() throws IOException {
        this.w = loadWeights();
    }

    /**
     * Processes the next frame of a stream.
     *
     * @param frame exactly {@link #FRAME_SIZE} signed 16-bit samples
     * @return speech probability in {@code [0, 1]}
     * @throws IllegalArgumentException when the frame doesn't have {@link #FRAME_SIZE} samples
     */
    public synchronized float process(short[] frame) {
        Objects.requireNonNull(frame, "frame must not be null");
        checkLength(frame.length);
        shiftContext();
        for (int i = 0; i < FRAME_SIZE; i++) {
            input[CONTEXT_SIZE + i] = frame[i] / 32768f;
        }
        return infer();
    }

    /**
     * Processes the next frame of a stream.
     *
     * @param frame exactly {@link #FRAME_SIZE} normalized floating-point audio samples in [-1.0,
     *     1.0]
     * @return speech probability in {@code [0, 1]}
     * @throws IllegalArgumentException when the frame doesn't have {@link #FRAME_SIZE} samples
     */
    public synchronized float process(float[] frame) {
        Objects.requireNonNull(frame, "frame must not be null");
        checkLength(frame.length);
        shiftContext();
        System.arraycopy(frame, 0, input, CONTEXT_SIZE, FRAME_SIZE);
        return infer();
    }

    /** Clears the stream state, so the instance can be used for a new stream. */
    public synchronized void reset() {
        Arrays.fill(input, 0f);
        Arrays.fill(h, 0f);
        Arrays.fill(c, 0f);
    }

    private static void checkLength(int length) {
        if (length != FRAME_SIZE) {
            throw new IllegalArgumentException(
                    "Expected " + FRAME_SIZE + " samples, got " + length);
        }
    }

    private void shiftContext() {
        System.arraycopy(input, FRAME_SIZE, input, 0, CONTEXT_SIZE);
    }

    /**
     * Executes neural network inference across STFT, Conv1D, LSTM, and linear layers.
     *
     * @return speech probability in {@code [0, 1]}
     */
    private float infer() {
        // Reflect pad the right side
        System.arraycopy(input, 0, padded, 0, input.length);
        for (int i = 0; i < PAD; i++) {
            padded[input.length + i] = input[input.length - 2 - i];
        }

        // STFT computation
        int frames = (padded.length - WINDOW) / HOP + 1;
        float[][] magnitude = new float[BINS][frames];
        for (int t = 0; t < frames; t++) {
            int offset = t * HOP;
            for (int bin = 0; bin < BINS; bin++) {
                float re = dot(w.stftBasis, bin * WINDOW, padded, offset, WINDOW);
                float im = dot(w.stftBasis, (bin + BINS) * WINDOW, padded, offset, WINDOW);
                magnitude[bin][t] = (float) Math.sqrt(re * re + im * im);
            }
        }

        // Conv1D computation + ReLU layers
        float[][] x = magnitude;
        for (int i = 0; i < 4; i++) {
            x = convRelu(x, w.convWeights[i], w.convBias[i], i == 1 || i == 2 ? 2 : 1);
        }

        float[] features = new float[HIDDEN];
        for (int i = 0; i < HIDDEN; i++) {
            features[i] = x[i][0];
        }

        // LSTM Gate computation
        for (int g = 0; g < 4 * HIDDEN; g++) {
            gates[g] =
                    w.lstmBias[g]
                            + dot(w.lstmInput, g * HIDDEN, features, 0, HIDDEN)
                            + dot(w.lstmHidden, g * HIDDEN, h, 0, HIDDEN);
        }
        // PyTorch gate order: input, forget, cell, output
        for (int i = 0; i < HIDDEN; i++) {
            c[i] =
                    sigmoid(gates[HIDDEN + i]) * c[i]
                            + sigmoid(gates[i]) * (float) Math.tanh(gates[2 * HIDDEN + i]);
            h[i] = sigmoid(gates[3 * HIDDEN + i]) * (float) Math.tanh(c[i]);
        }

        // Final linear classification layer + ReLU activation
        float out = w.outBias;
        for (int i = 0; i < HIDDEN; i++) {
            out += w.outWeight[i] * Math.max(h[i], 0f);
        }
        return sigmoid(out);
    }

    /**
     * Applies a 1D convolution with kernel size 3, symmetric padding 1, and ReLU activation.
     *
     * @param x input tensor of shape {@code [inChannels][inLength]}
     * @param weight flattened convolution kernel weights
     * @param bias bias vector per output channel
     * @param stride temporal stride (e.g., 1 or 2)
     */
    private static float[][] convRelu(float[][] x, float[] weight, float[] bias, int stride) {
        int inChannels = x.length;
        int inLength = x[0].length;
        int outChannels = bias.length;
        int outLength = (inLength - 1) / stride + 1;
        float[][] y = new float[outChannels][outLength];
        for (int o = 0; o < outChannels; o++) {
            for (int t = 0; t < outLength; t++) {
                float sum = bias[o];
                for (int i = 0; i < inChannels; i++) {
                    int base = (o * inChannels + i) * 3;
                    for (int k = 0; k < 3; k++) {
                        int pos = t * stride + k - 1;
                        if (pos >= 0 && pos < inLength) {
                            sum += weight[base + k] * x[i][pos];
                        }
                    }
                }
                y[o][t] = Math.max(sum, 0f);
            }
        }
        return y;
    }

    /**
     * Computes the vector dot product of two float arrays over a given range.
     *
     * @param a first vector
     * @param aOffset starting index in array {@code a}
     * @param b second vector
     * @param bOffset starting index in array {@code b}
     * @param length number of vector elements to accumulate
     * @return scalar dot product
     */
    private static float dot(float[] a, int aOffset, float[] b, int bOffset, int length) {
        float sum = 0f;
        for (int i = 0; i < length; i++) {
            sum += a[aOffset + i] * b[bOffset + i];
        }
        return sum;
    }

    /**
     * Computes the standard logistic sigmoid activation function: {@code 1 / (1 + e^{-v})}.
     *
     * @param v real input value
     * @return transformed value in the range {@code [0, 1]}
     */
    private static float sigmoid(float v) {
        return 1f / (1f + (float) Math.exp(-v));
    }

    private static synchronized Weights loadWeights() throws IOException {
        Weights weights = sharedWeights;
        if (weights == null) {
            try (InputStream in = SileroVAD.class.getResourceAsStream(WEIGHTS)) {
                if (in == null) {
                    throw new IOException("Missing " + WEIGHTS);
                }
                weights =
                        new Weights(
                                ByteBuffer.wrap(in.readAllBytes())
                                        .order(ByteOrder.LITTLE_ENDIAN)
                                        .asFloatBuffer());
            }
            sharedWeights = weights;
        }
        return weights;
    }

    /** Immutable container holding all model weights and biases. */
    private static class Weights {
        /**
         * Real and imaginary basis coefficients for the discrete STFT transformation in layout
         * {@code [2 * BINS * WINDOW]}.
         */
        final float[] stftBasis;

        /** Convolution kernel weights for the 4 convolutional layers. */
        final float[][] convWeights = new float[4][];

        /** Convolution bias vectors for the 4 convolutional layers. */
        final float[][] convBias = new float[4][];

        /** Weight matrix projecting input features to the 4 LSTM gates. */
        final float[] lstmInput;

        /** Recurrent weight matrix projecting previous hidden state to the 4 LSTM gates. */
        final float[] lstmHidden;

        /** Bias terms for the 4 LSTM gates. */
        final float[] lstmBias;

        /** Weight vector projecting the final activated hidden state to the output logit. */
        final float[] outWeight;

        /** Scalar bias term for the final output projection. */
        final float outBias;

        Weights(FloatBuffer buffer) throws IOException {
            stftBasis = read(buffer, 2 * BINS * WINDOW);
            int[][] shapes = {{128, 129}, {64, 128}, {64, 64}, {128, 64}};
            for (int i = 0; i < 4; i++) {
                convWeights[i] = read(buffer, shapes[i][0] * shapes[i][1] * 3);
                convBias[i] = read(buffer, shapes[i][0]);
            }
            lstmInput = read(buffer, 4 * HIDDEN * HIDDEN);
            lstmHidden = read(buffer, 4 * HIDDEN * HIDDEN);
            lstmBias = read(buffer, 4 * HIDDEN);
            outWeight = read(buffer, HIDDEN);
            outBias = read(buffer, 1)[0];
            if (buffer.hasRemaining()) {
                throw new IOException("Unexpected Silero weights size");
            }
        }

        private static float[] read(FloatBuffer buffer, int size) throws IOException {
            if (buffer.remaining() < size) {
                throw new IOException("Unexpected Silero weights size");
            }
            float[] values = new float[size];
            buffer.get(values);
            return values;
        }
    }
}
