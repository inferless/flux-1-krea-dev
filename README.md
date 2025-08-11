# Model Template - FLUX.1-Krea-dev
[FLUX.1-Krea-dev](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev) is an open-weights text-to-image model from Black Forest Labs, developed in collaboration with Krea. It’s a 12-billion-parameter rectified-flow transformer distilled from Krea 1, designed to deliver highly photorealistic results with a distinctive, “opinionated” aesthetic that avoids the oversaturated “AI look.” Fully compatible with the FLUX.1-[dev] ecosystem, it defines a new state-of-the-art in prompt adherence, style diversity, and scene complexity while emphasizing realism and pleasing, varied visuals.

### Fork the Repository
Get started by forking the repository. You can do this by clicking on the fork button in the top right corner of the repository page.

This will create a copy of the repository in your own GitHub account, allowing you to make changes and customize it according to your needs.

### Create a Custom Runtime in Inferless
To access the custom runtime window in Inferless, simply navigate to the sidebar and click on the Create new Runtime button. A pop-up will appear.

Next, provide a suitable name for your custom runtime and proceed by uploading the **inferless-runtime-config.yaml** file given above. Finally, ensure you save your changes by clicking on the save button.

### Import the Model in Inferless
Log in to your inferless account, select the workspace you want the model to be imported into and click the Add Model button.

Select the PyTorch as framework and choose **Repo(custom code)** as your model source and select your provider, and use the forked repo URL as the **Model URL**.

Enter all the required details to Import your model. Refer [this link](https://docs.inferless.com/integrations/git-custom-code/git--custom-code) for more information on model import.

### Add Your Hugging Face Access Token
This model requires a Hugging Face access token for authentication. You can provide the token in the following ways:

- **Via the Platform UI**: Set the `HF_TOKEN` in the **Environment Variables** section.
- **Via the CLI**: Add the `HF_TOKEN` as an environment variable.

---
## Curl Command
Following is an example of the curl command you can use to make inference. You can find the exact curl command in the Model's API page in Inferless.
```bash
curl --location '<your_inference_url>' \
    --header 'Content-Type: application/json' \
    --header 'Authorization: Bearer <your_api_key>' \
    --data '{
              {
                "inputs": [
                  {
                    "name": "prompt",
                    "shape": [1],
                    "data": ["A cat holding a sign that says hello"],
                    "datatype": "BYTES"
                  },
                  {
                    "name": "height",
                    "optional": true,
                    "shape": [1],
                    "data": [1024],
                    "datatype": "INT32"
                  },
                  {
                    "name": "width",
                    "optional": true,
                    "shape": [1],
                    "data": [1024],
                    "datatype": "INT32"
                  }
                ]
              }

    }'
```

---
## Customizing the Code
Open the `app.py` file. This contains the main code for inference. It has three main functions, initialize, infer and finalize.

**Initialize** -  This function is executed during the cold start and is used to initialize the model. If you have any custom configurations or settings that need to be applied during the initialization, make sure to add them in this function.

**Infer** - This function is where the inference happens. The argument to this function `inputs`, is a dictionary containing all the input parameters. The keys are the same as the name given in inputs. Refer to [input](https://docs.inferless.com/model-import/input-output-schema) for more.

**Finalize** - This function is used to perform any cleanup activity for example you can unload the model from the gpu by setting to `None`.

For more information refer to the [Inferless docs](https://docs.inferless.com/).
