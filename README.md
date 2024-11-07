# Luigi Goes Brrr: Simplifying Task Orchestration Without the Headaches

This repository accompanies the presentation, [**"Luigi Goes Brrr: Simplifying Task Orchestration Without the Headaches."**](https://docs.google.com/presentation/d/1HLeYnUtLSBJdShBBf0SG5bSbujdwY9kb8Ia6bWaigzE/edit?usp=sharing)
It demonstrates how to use [Luigi](https://github.com/spotify/luigi), an open-source Python tool for orchestrating data workflows.

Luigi provides a simple yet powerful approach to handling dependencies and running complex pipelines without the need for heavier orchestration tools like Airflow, Prefect, or Dagster.

## Overview

In this example, we build a complete data pipeline using Luigi, showcasing Luigi’s ability to manage dependencies, run parameterized tasks, and handle workflows efficiently:

1. **Data Ingestion**: Loads the `iris` dataset.
2. **Data Normalization**: Scales features for consistent model input.
3. **Train/Test Split**: Splits the data and labels rows as train or test.
4. **Save Train/Test Data**: Saves separate train and test datasets.
5. **Model Training**: Trains a logistic regression model on the training data.
6. **Model Validation**: Validates the model on the test data and logs accuracy.

This pipeline is ideal for showcasing the simplicity and flexibility of Luigi for batch data processing tasks.

## Running the Pipeline

### Using the Makefile

The repository includes a `Makefile` for streamlined commands to format, lint, clean, and run the pipeline.

1. **Run the Full Pipeline**:
    ```bash
    make run
    ```

2. **Format the Code**:
    ```bash
    make format
    ```

3. **Lint the Code**:
    ```bash
    make lint
    ```

4. **Clean Up Output Directories**:
    ```bash
    make clean
    ```

## Key Takeaways

- **Simplicity**: Luigi provides a lightweight, open-source solution for orchestrating complex data workflows without the need for paid features or heavy infrastructure.
- **Dependency Management**: Luigi’s task-based structure makes managing complex dependencies and running sequential tasks straightforward.
- **Flexibility and Control**: Fully open-source, Luigi allows for flexible parameterization and gives you full control over task execution without vendor lock-in.
