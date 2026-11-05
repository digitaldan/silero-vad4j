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
import java.nio.charset.StandardCharsets;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

public class SileroVADTest {

    @Test
    public void matchesOnnxModel() throws IOException {
        short[] audio = speechWithTrailingSilence();
        float[] expected = expectedProbabilities();
        SileroVAD vad = new SileroVAD();

        assertEquals(expected.length, audio.length / SileroVAD.FRAME_SIZE);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], vad.process(frame(audio, i)), 1e-4, "frame " + i);
        }
    }

    @Test
    public void floatFramesMatchShortFrames() throws IOException {
        short[] audio = speechWithTrailingSilence();
        SileroVAD shorts = new SileroVAD();
        SileroVAD floats = new SileroVAD();

        for (int i = 0; i < audio.length / SileroVAD.FRAME_SIZE; i++) {
            short[] frame = frame(audio, i);
            float[] scaled = new float[frame.length];
            for (int j = 0; j < frame.length; j++) {
                scaled[j] = frame[j] / 32768f;
            }
            assertEquals(shorts.process(frame), floats.process(scaled), "frame " + i);
        }
    }

    @Test
    public void resetStartsANewStream() throws IOException {
        short[] audio = speechWithTrailingSilence();
        float[] expected = expectedProbabilities();
        SileroVAD vad = new SileroVAD();

        for (int i = 0; i < 20; i++) {
            vad.process(frame(audio, i));
        }
        vad.reset();
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], vad.process(frame(audio, i)), 1e-4, "frame " + i);
        }
    }

    @Test
    public void noiseIsNotSpeech() throws IOException {
        Random random = new Random(1);
        short[] audio = new short[SileroVAD.SAMPLE_RATE];
        for (int i = 0; i < audio.length; i++) {
            audio[i] = (short) (random.nextGaussian() * 1500);
        }
        SileroVAD vad = new SileroVAD();
        for (int i = 0; i < audio.length / SileroVAD.FRAME_SIZE; i++) {
            assertTrue(vad.process(frame(audio, i)) < 0.5f, "frame " + i);
        }
    }

    @Test
    public void rejectsWrongFrameSize() throws IOException {
        SileroVAD vad = new SileroVAD();
        assertThrows(IllegalArgumentException.class, () -> vad.process(new short[256]));
        assertThrows(IllegalArgumentException.class, () -> vad.process(new float[1024]));
    }

    private static short[] frame(short[] audio, int index) {
        short[] frame = new short[SileroVAD.FRAME_SIZE];
        System.arraycopy(audio, index * SileroVAD.FRAME_SIZE, frame, 0, frame.length);
        return frame;
    }

    /**
     * "Turn on the kitchen lights." synthesized by Piper TTS with the en_US-ljspeech-medium voice
     * (MIT, trained on the public domain LJ Speech dataset), resampled to 16 kHz, followed by 8000
     * silent samples. speech-silero-probs.txt holds the official ONNX model output for this audio,
     * see scripts/export_weights.py.
     */
    private static short[] speechWithTrailingSilence() throws IOException {
        byte[] wav = bytes("/speech.wav");
        ByteBuffer buffer = ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN);
        int offset = 12;
        while (!"data".equals(new String(wav, offset, 4, StandardCharsets.US_ASCII))) {
            offset += 8 + buffer.getInt(offset + 4);
        }
        int samples = buffer.getInt(offset + 4) / 2;
        short[] audio = new short[samples + 8000];
        buffer.position(offset + 8);
        buffer.asShortBuffer().get(audio, 0, samples);
        return audio;
    }

    private static float[] expectedProbabilities() throws IOException {
        String[] lines =
                new String(bytes("/speech-silero-probs.txt"), StandardCharsets.UTF_8)
                        .trim()
                        .split("\n");
        float[] values = new float[lines.length];
        for (int i = 0; i < lines.length; i++) {
            values[i] = Float.parseFloat(lines[i]);
        }
        return values;
    }

    private static byte[] bytes(String name) throws IOException {
        try (InputStream in = SileroVADTest.class.getResourceAsStream(name)) {
            assertNotNull(in, name);
            return in.readAllBytes();
        }
    }
}
