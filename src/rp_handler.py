import runpod
from runpod.serverless.utils import rp_upload
import json
import urllib.request
import urllib.parse
import time
import os
import requests
import base64
import random
from io import BytesIO

# Time to wait between API check attempts in milliseconds
COMFY_API_AVAILABLE_INTERVAL_MS = 50
# Maximum number of API check attempts
COMFY_API_AVAILABLE_MAX_RETRIES = 500
# Time to wait between poll attempts in milliseconds
COMFY_POLLING_INTERVAL_MS = int(os.environ.get("COMFY_POLLING_INTERVAL_MS", 250))
# Maximum number of poll attempts
COMFY_POLLING_MAX_RETRIES = int(os.environ.get("COMFY_POLLING_MAX_RETRIES", 500))
# Host where ComfyUI is running
COMFY_HOST = "127.0.0.1:8188"
# Enforce a clean state after each job is done
# see https://docs.runpod.io/docs/handler-additional-controls#refresh-worker
REFRESH_WORKER = os.environ.get("REFRESH_WORKER", "false").lower() == "true"

# Path to the workflow JSON file
WORKFLOW_FILE_PATH = "SkimProdPhotoshoot-Serverless-workflow-API-format.json"


def load_workflow_template():
    """
    Loads the workflow template from the JSON file.
    
    Returns:
        dict: The workflow template loaded from JSON file
    """
    try:
        with open(WORKFLOW_FILE_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Workflow file not found: {WORKFLOW_FILE_PATH}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in workflow file: {e}")
    except Exception as e:
        raise Exception(f"Error loading workflow file: {e}")


def construct_workflow(prompt, height, width, filename):
    """
    Constructs the ComfyUI workflow by loading the template and replacing parameters.
    
    Args:
        prompt (str): The text prompt for image generation
        height (int): The height of the output image
        width (int): The width of the output image
        filename (str): The filename of the input image
        
    Returns:
        dict: The constructed workflow with parameters replaced
    """
    import copy
    
    # Load the workflow template from JSON file
    workflow = load_workflow_template()
    
    # Deep copy to avoid modifying the original
    workflow = copy.deepcopy(workflow)
    
    # Replace prompt in node 583
    workflow["583"]["inputs"]["text"] = prompt
    
    # Replace width and height in node 7
    workflow["7"]["inputs"]["width"] = width
    workflow["7"]["inputs"]["height"] = height
    
    # Replace filename in node 1 (LoadImage)
    workflow["1"]["inputs"]["image"] = filename
    
    # Generate random seed for node 597
    random_seed = random.randint(1, 999999999999999)
    workflow["597"]["inputs"]["seed"] = random_seed
    
    return workflow


def download_image_from_s3(image_url):
    """
    Downloads an image from S3 URL and converts it to base64.
    
    Args:
        image_url (str): The S3 URL of the image to download
        
    Returns:
        tuple: (filename, base64_image_data) or (None, error_message)
    """
    try:
        # Download the image from S3
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        # Get the filename from the URL
        filename = image_url.split('/')[-1]
        if not filename or '.' not in filename:
            filename = f"input_image_{int(time.time())}.jpg"
        
        # Convert to base64
        image_data = base64.b64encode(response.content).decode('utf-8')
        
        return filename, image_data
        
    except requests.RequestException as e:
        return None, f"Failed to download image from S3: {str(e)}"
    except Exception as e:
        return None, f"Error processing image: {str(e)}"


def validate_input(job_input):
    """
    Validates the input for the handler function.

    Args:
        job_input (dict): The input data to validate.

    Returns:
        tuple: A tuple containing the validated data and an error message, if any.
               The structure is (validated_data, error_message).
    """
    # Validate if job_input is provided
    if job_input is None:
        return None, "Please provide input"

    # Check if input is a string and try to parse it as JSON
    if isinstance(job_input, str):
        try:
            job_input = json.loads(job_input)
        except json.JSONDecodeError:
            return None, "Invalid JSON format in input"

    # Validate required parameters
    prompt = job_input.get("prompt")
    if prompt is None:
        return None, "Missing 'prompt' parameter"
    
    height = job_input.get("height")
    if height is None:
        return None, "Missing 'height' parameter"
    
    width = job_input.get("width")
    if width is None:
        return None, "Missing 'width' parameter"
    
    image_url = job_input.get("image_url")
    if image_url is None:
        return None, "Missing 'image_url' parameter"
    
    # Validate data types
    if not isinstance(prompt, str):
        return None, "'prompt' must be a string"
    
    if not isinstance(height, (int, float)) or height <= 0:
        return None, "'height' must be a positive number"
    
    if not isinstance(width, (int, float)) or width <= 0:
        return None, "'width' must be a positive number"
    
    if not isinstance(image_url, str):
        return None, "'image_url' must be a string"

    # Return validated data and no error
    return {
        "prompt": prompt,
        "height": int(height),
        "width": int(width),
        "image_url": image_url
    }, None


def check_server(url, retries=500, delay=50):
    """
    Check if a server is reachable via HTTP GET request

    Args:
    - url (str): The URL to check
    - retries (int, optional): The number of times to attempt connecting to the server. Default is 50
    - delay (int, optional): The time in milliseconds to wait between retries. Default is 500

    Returns:
    bool: True if the server is reachable within the given number of retries, otherwise False
    """

    for i in range(retries):
        try:
            response = requests.get(url)

            # If the response status code is 200, the server is up and running
            if response.status_code == 200:
                print(f"runpod-worker-comfy - API is reachable")
                return True
        except requests.RequestException as e:
            # If an exception occurs, the server may not be ready
            pass

        # Wait for the specified delay before retrying
        time.sleep(delay / 1000)

    print(
        f"runpod-worker-comfy - Failed to connect to server at {url} after {retries} attempts."
    )
    return False


def upload_images(images):
    """
    Upload a list of base64 encoded images to the ComfyUI server using the /upload/image endpoint.

    Args:
        images (list): A list of dictionaries, each containing the 'name' of the image and the 'image' as a base64 encoded string.
        server_address (str): The address of the ComfyUI server.

    Returns:
        list: A list of responses from the server for each image upload.
    """
    if not images:
        return {"status": "success", "message": "No images to upload", "details": []}

    responses = []
    upload_errors = []

    print(f"runpod-worker-comfy - image(s) upload")

    for image in images:
        name = image["name"]
        image_data = image["image"]
        blob = base64.b64decode(image_data)

        # Prepare the form data
        files = {
            "image": (name, BytesIO(blob), "image/png"),
            "overwrite": (None, "true"),
        }

        # POST request to upload the image
        response = requests.post(f"http://{COMFY_HOST}/upload/image", files=files)
        if response.status_code != 200:
            upload_errors.append(f"Error uploading {name}: {response.text}")
        else:
            responses.append(f"Successfully uploaded {name}")

    if upload_errors:
        print(f"runpod-worker-comfy - image(s) upload with errors")
        return {
            "status": "error",
            "message": "Some images failed to upload",
            "details": upload_errors,
        }

    print(f"runpod-worker-comfy - image(s) upload complete")
    return {
        "status": "success",
        "message": "All images uploaded successfully",
        "details": responses,
    }


def queue_workflow(workflow):
    """
    Queue a workflow to be processed by ComfyUI

    Args:
        workflow (dict): A dictionary containing the workflow to be processed

    Returns:
        dict: The JSON response from ComfyUI after processing the workflow
    """

    # The top level element "prompt" is required by ComfyUI
    data = json.dumps({"prompt": workflow}).encode("utf-8")

    req = urllib.request.Request(f"http://{COMFY_HOST}/prompt", data=data)
    return json.loads(urllib.request.urlopen(req).read())


def get_history(prompt_id):
    """
    Retrieve the history of a given prompt using its ID

    Args:
        prompt_id (str): The ID of the prompt whose history is to be retrieved

    Returns:
        dict: The history of the prompt, containing all the processing steps and results
    """
    with urllib.request.urlopen(f"http://{COMFY_HOST}/history/{prompt_id}") as response:
        return json.loads(response.read())


def base64_encode(img_path):
    """
    Returns base64 encoded image.

    Args:
        img_path (str): The path to the image

    Returns:
        str: The base64 encoded image
    """
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return f"{encoded_string}"


def process_output_images(outputs, job_id):
    """
    This function takes the "outputs" from image generation and the job ID,
    then determines the correct way to return the image, either as a direct URL
    to an AWS S3 bucket or as a base64 encoded string, depending on the
    environment configuration.

    Args:
        outputs (dict): A dictionary containing the outputs from image generation,
                        typically includes node IDs and their respective output data.
        job_id (str): The unique identifier for the job.

    Returns:
        dict: A dictionary with the status ('success' or 'error') and the message,
              which is either the URL to the image in the AWS S3 bucket or a base64
              encoded string of the image. In case of error, the message details the issue.

    The function works as follows:
    - It first determines the output path for the images from an environment variable,
      defaulting to "/comfyui/output" if not set.
    - It then looks specifically for the final output node (740) which contains the final generated image.
    - After confirming the existence of the image in the output folder, it checks if the
      AWS S3 bucket is configured via the BUCKET_ENDPOINT_URL environment variable.
    - If AWS S3 is configured, it uploads the image to the bucket and returns the URL.
    - If AWS S3 is not configured, it encodes the image in base64 and returns the string.
    - If the image file does not exist in the output folder, it returns an error status
      with a message indicating the missing image file.
    """

    # The path where ComfyUI stores the generated images
    COMFY_OUTPUT_PATH = os.environ.get("COMFY_OUTPUT_PATH", "/comfyui/output")

    # DEBUG: Log the entire outputs structure
    print(f"runpod-worker-comfy - DEBUG: Full outputs structure:")
    print(f"runpod-worker-comfy - DEBUG: Outputs keys: {list(outputs.keys())}")
    
    # DEBUG: Log each output node that contains images
    for node_id, node_output in outputs.items():
        if "images" in node_output:
            print(f"runpod-worker-comfy - DEBUG: Node {node_id} has {len(node_output['images'])} images")
            for i, image in enumerate(node_output["images"]):
                print(f"runpod-worker-comfy - DEBUG:   Image {i}: subfolder='{image.get('subfolder', 'N/A')}', filename='{image.get('filename', 'N/A')}'")
        else:
            print(f"runpod-worker-comfy - DEBUG: Node {node_id} has no images")

    # Look specifically for the final output node (740 - "ImageCompositeMasked (Final image output)")
    final_output_node = "740"
    
    if final_output_node not in outputs:
        print(f"runpod-worker-comfy - DEBUG: Node {final_output_node} not found in outputs")
        # Try to find any node with images as fallback
        nodes_with_images = [node_id for node_id, node_output in outputs.items() if "images" in node_output]
        if nodes_with_images:
            print(f"runpod-worker-comfy - DEBUG: Found nodes with images: {nodes_with_images}")
            # Use the first available node with images
            final_output_node = nodes_with_images[0]
            print(f"runpod-worker-comfy - DEBUG: Using fallback node {final_output_node}")
        else:
            return {
                "status": "error",
                "message": f"Final output node {final_output_node} not found in outputs, and no other nodes with images found",
            }
    
    if "images" not in outputs[final_output_node]:
        return {
            "status": "error",
            "message": f"Node {final_output_node} found but has no images",
        }

    # Get the first image from the final output node
    final_images = outputs[final_output_node]["images"]
    if not final_images:
        return {
            "status": "error",
            "message": f"No images found in final output node {final_output_node}",
        }

    # Construct the image path
    image_info = final_images[0]  # Get the first image
    image_path = os.path.join(image_info["subfolder"], image_info["filename"])
    local_image_path = f"{COMFY_OUTPUT_PATH}/{image_path}"

    print(f"runpod-worker-comfy - image generation is done")
    print(f"runpod-worker-comfy - {local_image_path}")

    # The image is in the output folder
    if os.path.exists(local_image_path):
        if os.environ.get("BUCKET_ENDPOINT_URL", False):
            # URL to image in AWS S3
            image = rp_upload.upload_image(job_id, local_image_path)
            print(
                "runpod-worker-comfy - the image was generated and uploaded to AWS S3"
            )
        else:
            # base64 image
            image = base64_encode(local_image_path)
            print(
                "runpod-worker-comfy - the image was generated and converted to base64"
            )

        return {
            "status": "success",
            "message": image,
        }
    else:
        print("runpod-worker-comfy - the image does not exist in the output folder")
        return {
            "status": "error",
            "message": f"the image does not exist in the specified output folder: {local_image_path}",
        }


def handler(job):
    """
    The main function that handles a job of generating an image.

    This function validates the input, downloads the image from S3, constructs the workflow,
    sends a prompt to ComfyUI for processing, polls ComfyUI for result, and retrieves generated images.

    Args:
        job (dict): A dictionary containing job details and input parameters.

    Returns:
        dict: A dictionary containing either an error message or a success status with generated images.
    """
    job_input = job["input"]

    # Make sure that the input is valid
    validated_data, error_message = validate_input(job_input)
    if error_message:
        return {"error": error_message}

    # Extract validated data
    prompt = validated_data["prompt"]
    height = validated_data["height"]
    width = validated_data["width"]
    image_url = validated_data["image_url"]

    # Download image from S3
    filename, image_data = download_image_from_s3(image_url)
    if filename is None:
        return {"error": f"Failed to download image from S3: {image_data}"}

    # Make sure that the ComfyUI API is available
    check_server(
        f"http://{COMFY_HOST}",
        COMFY_API_AVAILABLE_MAX_RETRIES,
        COMFY_API_AVAILABLE_INTERVAL_MS,
    )

    # Upload the downloaded image to ComfyUI
    upload_result = upload_images([{"name": filename, "image": image_data}])
    if upload_result["status"] == "error":
        return upload_result

    # Construct the workflow with user parameters
    workflow = construct_workflow(prompt, height, width, filename)

    # Queue the workflow
    try:
        queued_workflow = queue_workflow(workflow)
        prompt_id = queued_workflow["prompt_id"]
        print(f"runpod-worker-comfy - queued workflow with ID {prompt_id}")
    except Exception as e:
        return {"error": f"Error queuing workflow: {str(e)}"}

    # Poll for completion
    print(f"runpod-worker-comfy - wait until image generation is complete")
    retries = 0
    try:
        while retries < COMFY_POLLING_MAX_RETRIES:
            history = get_history(prompt_id)

            # Exit the loop if we have found the history
            if prompt_id in history and history[prompt_id].get("outputs"):
                break
            else:
                # Wait before trying again
                time.sleep(COMFY_POLLING_INTERVAL_MS / 1000)
                retries += 1
        else:
            return {"error": "Max retries reached while waiting for image generation"}
    except Exception as e:
        return {"error": f"Error waiting for image generation: {str(e)}"}

    # Get the generated image and return it as URL in an AWS bucket or as base64
    images_result = process_output_images(history[prompt_id].get("outputs"), job["id"])

    result = {**images_result, "refresh_worker": REFRESH_WORKER}

    return result


# Start the handler only if this script is run directly
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
