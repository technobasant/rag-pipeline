# JAR Files for Spark/Hadoop Integration

## Overview
This project requires large JAR files for Spark and Hadoop integration, but these files are excluded from Git version control due to GitHub's 100 MB file size limit.

## Required JAR Files
The following JAR files should be placed in the `jars/` directory for full ETL functionality:

- `aws-java-sdk-bundle-1.12.262.jar` (267.64 MB)
- `hadoop-aws-3.3.4.jar`
- `hadoop-common-3.3.4.jar`

## Download Instructions

### Option 1: Maven Central Repository
Download the JAR files from Maven Central:

```bash
# Create jars directory
mkdir -p jars/

# Download AWS Java SDK Bundle
curl -o jars/aws-java-sdk-bundle-1.12.262.jar \
  https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar

# Download Hadoop AWS
curl -o jars/hadoop-aws-3.3.4.jar \
  https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar

# Download Hadoop Common
curl -o jars/hadoop-common-3.3.4.jar \
  https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-common/3.3.4/hadoop-common-3.3.4.jar
```

### Option 2: Using Maven/Gradle
If you have Maven installed, you can use the dependency plugin:

```bash
mvn dependency:copy -Dartifact=com.amazonaws:aws-java-sdk-bundle:1.12.262 -DoutputDirectory=jars/
mvn dependency:copy -Dartifact=org.apache.hadoop:hadoop-aws:3.3.4 -DoutputDirectory=jars/
mvn dependency:copy -Dartifact=org.apache.hadoop:hadoop-common:3.3.4 -DoutputDirectory=jars/
```

## Why These Files Are Excluded

1. **GitHub File Size Limit**: GitHub has a 100 MB file size limit per file
2. **Repository Size**: Large binary files significantly increase repository clone time
3. **Version Control**: JAR files are build artifacts and don't need version history
4. **Availability**: These files are publicly available from Maven Central

## Development Impact

- **Tests**: ETL tests are skipped when JAR files are not available
- **Docker**: The Dockerfile handles missing JARs gracefully
- **Local Development**: Download JARs only when working on ETL features
- **CI/CD**: Build pipelines can download JARs as needed

## Verification

After downloading the JAR files, verify they're in place:

```bash
ls -la jars/
# Should show:
# aws-java-sdk-bundle-1.12.262.jar
# hadoop-aws-3.3.4.jar  
# hadoop-common-3.3.4.jar
```

## Alternative: Docker-based Development

For development without local JAR files, use the Docker environment:

```bash
# All dependencies are handled in the container
docker-compose up -d
```

The Docker containers will handle Spark/Hadoop dependencies automatically.