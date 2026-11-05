# silero-vad4j - Publish

Create a new release:

```shell
./mvnw release:prepare
./mvnw release:clean
```

Push the release commits and tag to GitHub:

```shell
git push origin main --follow-tags
```

GitHub Actions will publish the artifacts to Maven Central.
