# LLM Voice Assistant Framework

This is the code to the paper "LLM Voice Assistant Framework: Towards the Next Generation of Personal Assistants".

## Prerequisites

The following tools need to be installed to build and run the service locally:

- Python 3.11.5 (3.12. or higher might not be supported by some libraries)

## Run the application

(Mac: run `brew install portaudio`)

1. Setup a venv environment: `python3.11 -m venv .venv` and `source .venv/bin/activate`
2. Install dependencies: `python -m pip install -r requirements.txt`
3. Register at Google Cloud and create a new project. For that project, setup and enable [Billing](https://console.cloud.google.com/billing/) and the [Speech Services](https://console.cloud.google.com/speech).
4. Install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install-sdk). Run `gcloud init` and/or `gcloud auth login` to authenticate, select the created project.
5. Setup an AWS account and configure your [AWS credentials](https://github.com/boto/boto3?tab=readme-ov-file#using-boto3).
6. Create a `.env` file in the root directory by copying the `.env.template` file and adding your secrets.
7. Run the app: `python src/main.py`. Strictly ensure to run the app from the root directory.

## Contributing

For any questions or issues regarding the code, please refer to the README or open an issue.
