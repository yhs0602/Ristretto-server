# Ristretto-3B

## Run the server
python server.py

## Client (text only)
### Request
```sh
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"LiAutoAD/Ristretto-3B",
    "messages":[
      {"role":"user","content":"Hello, who are you?"}
    ]
  }'
```

### Response
```json
{
   "id":"chatcmpl-1747723424",
   "object":"chat.completion",
   "created":1747723424,
   "model":"LiAutoAD/Ristretto-3B",
   "choices":[
      {
         "index":0,
         "message":{
            "role":"assistant",
            "content":"Hello! I am an AI assistant created by OpenAI. My purpose is to help you with a wide range of tasks, answer your questions, and provide information on various topics. How can I assist you today?"
         },
         "finish_reason":"stop"
      }
   ],
   "usage":{
      "prompt_tokens":6,
      "completion_tokens":43,
      "total_tokens":49
   }
}
```


## Client (text with image)
### Request
```sh
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"LiAutoAD/Ristretto-3B",
    "messages":[
      {"role":"user","content":{"image_url":"https://your.domain/image.png"}},
      {"role":"user","content":"Please describe the image."}
    ]
  }'
```

### Response
```json
{
   "id":"chatcmpl-1747723512",
   "object":"chat.completion",
   "created":1747723512,
   "model":"LiAutoAD/Ristretto-3B",
   "choices":[
      {
         "index":0,
         "message":{
            "role":"assistant",
            "content":"The image features a golden retriever dog lying on the ground in a relaxed position. The dog has a fluffy, golden coat and is panting with its tongue out, suggesting it is either tired or enjoying the outdoors. The background is blurred, but it appears to be a natural setting with trees and foliage, indicating that the dog is likely in a park or forested area. The ground is covered with leaves, further supporting the idea that this is an outdoor scene. The lighting in the image is soft, creating a warm and inviting atmosphere."
         },
         "finish_reason":"stop"
      }
   ],
   "usage":{
      "prompt_tokens":8,
      "completion_tokens":109,
      "total_tokens":117
   }
}
```