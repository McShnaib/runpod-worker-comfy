# API Changes Summary

## Overview
The `rp_handler.py` has been modified to accept simplified parameters instead of requiring a full ComfyUI workflow. The handler now automatically constructs the workflow based on your input parameters.

## What Changed

### 1. **New Input Format**
Instead of sending a complex workflow, your webapp now sends:

```json
{
  "input": {
    "prompt": "Your text prompt here",
    "height": 904,
    "width": 1600,
    "image_s3_url": "https://your-bucket.s3.region.amazonaws.com/image.jpg"
  }
}
```

### 2. **Automatic Workflow Construction**
The handler now:
- **Constructs the workflow** automatically using your parameters
- **Downloads the image** from your S3 URL
- **Generates a random seed** for each request
- **Replaces placeholders** in the workflow template

### 3. **Parameter Mapping**
- `prompt` → Node 583 (CR Text - prompt parameter)
- `height` → Nodes 7 and 97 (Image size parameters)
- `width` → Nodes 7 and 97 (Image size parameters)
- `image_s3_url` → Node 1 (Load Image)
- `random_seed` → Node 597 (KSampler) - generated automatically

## Benefits

1. **Simpler API**: No need to understand ComfyUI workflow structure
2. **Consistent Results**: Same workflow template ensures consistent processing
3. **Easy Integration**: Your webapp just sends basic parameters
4. **Maintainable**: Workflow logic is centralized in the handler
5. **Flexible**: Easy to modify the workflow template for different use cases

## How It Works

1. **Receive Request**: Handler gets your 4 parameters
2. **Download Image**: Downloads image from S3 URL
3. **Construct Workflow**: Builds the complete ComfyUI workflow
4. **Execute**: Runs the workflow through ComfyUI
5. **Return Result**: Returns generated image (base64 or S3 URL)

## Example Usage

```bash
curl -X POST -H "Authorization: Bearer <api_key>" \
  -H "Content-Type: application/json" \
  -d @test_new_api.json \
  https://api.runpod.ai/v2/<endpoint_id>/runsync
```

## Backward Compatibility

⚠️ **Breaking Change**: The old API format (workflow + images) is no longer supported. All existing integrations will need to be updated to use the new simplified format.

## Files Modified

- `src/rp_handler.py` - Main handler with new workflow construction logic
- `test_new_api.json` - Example of new API format
- `API_CHANGES_SUMMARY.md` - This documentation

## Next Steps

1. **Update your webapp** to send the new parameter format
2. **Test the new API** with the provided test file
3. **Deploy the updated handler** to your RunPod endpoint
4. **Monitor logs** to ensure proper workflow execution
