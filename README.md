# Silero VAD for Java

[![Maven Central](https://img.shields.io/maven-central/v/io.github.jvoice-project/silero-vad4j.svg?label=Maven%20Central)](https://central.sonatype.com/artifact/io.github.jvoice-project/silero-vad4j)
[![CI](https://github.com/jvoice-project/silero-vad4j/actions/workflows/build.yml/badge.svg)](https://github.com/jvoice-project/silero-vad4j/actions/workflows/build.yml)
[![License](https://img.shields.io/github/license/jvoice-project/silero-vad4j.svg)](LICENSE)

A plain Java inference library for [Silero VAD](https://github.com/snakers4/silero-vad), a pre-trained voice activity detector.

Silero VAD is a small neural network that returns the probability that a 32 ms frame of audio contains speech.
It has a high level of accuracy even in noisy environments.

This library runs the model in plain Java with the official weights bundled.
It has no dependencies and no native code, so it runs on any platform with Java >= 21.

Only 16 kHz mono audio is supported.

## Usage

The package is distributed through [Maven Central](https://central.sonatype.com/artifact/io.github.jvoice-project/silero-vad4j).

Version numbers consist of four parts separated by dots:

1. The first three parts are the version of the Silero VAD model the weights are exported from.
1. The last part is an incremental revision of this library.

### Maven

```xml
<dependency>
    <groupId>io.github.jvoice-project</groupId>
    <artifactId>silero-vad4j</artifactId>
    <!-- replace $version with a specific version -->
    <version>$version</version>
</dependency>
```

### Gradle

```gradle
repositories {
    mavenCentral()
}

dependencies {
    implementation 'io.github.jvoice-project:silero-vad4j:+' // gets the latest version
}
```

### Example

```java
SileroVAD vad = new SileroVAD();

short[] samples = ...; // 16 kHz, 16-bit mono PCM samples
int step = SileroVAD.FRAME_SIZE; // 512 samples (32 ms), the only supported frame size

for (int i = 0; i + step <= samples.length; i += step) {
    short[] frame = Arrays.copyOfRange(samples, i, i + step);
    if (vad.process(frame) >= 0.5f) {
        System.out.println("Voice detected at sample index: " + i);
        break;
    }
}
```

`process` also accepts `float[]` frames with samples between -1 and 1.

The model keeps state between frames, so use one instance per audio stream and feed it consecutive frames.
Call `reset()` before reusing an instance for a new stream.
The weights (about 1.2 MB) are loaded once and shared by all instances.

A threshold of 0.5 is a good default.
Raise it to reject more background noise, lower it to pick up quiet speakers.

## Model Weights

`src/main/resources/io/github/jvoiceproject/silerovad/silero_vad_16k.bin` holds the 16 kHz weights of the official ONNX model as little-endian floats.
See [NOTICE](NOTICE) for the source version and checksum.

To update the weights, download `silero_vad.onnx` from the [Silero VAD repository](https://github.com/snakers4/silero-vad/tree/master/src/silero_vad/data) and run:

```shell
pip install onnx onnxruntime numpy
python scripts/export_weights.py silero_vad.onnx
```

This writes the weights and regenerates the ONNX reference output that the tests compare against.
Then update the version in [NOTICE](NOTICE), the Javadoc of `SileroVAD` and the version number in `pom.xml`.
The script only supports the v5 architecture, which v6 still uses.

## Development

You need Java >= 21.

Build the library, run the tests, and package the JAR with Maven:

```shell
./mvnw package
```

Run code formatting checking (`./mvnw spotless:check`) before submitting changes.
