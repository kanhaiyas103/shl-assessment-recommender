# API Documentation

The public API intentionally exposes exactly two endpoints required by the assignment.
OpenAPI docs are disabled in production to keep the submission surface minimal.

## `GET /health`

Returns service health for local checks and Render.

### Response

HTTP `200`

```json
{
  "status": "ok"
}
```

No additional fields are returned.

## `POST /chat`

Runs one stateless conversation turn. The request must include the complete ordered
conversation history. The latest message must be from the user.

### Request schema

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hiring a Java developer"
    }
  ]
}
```

Rules:

- `messages` must contain 1 to 8 messages.
- Roles must be `user` or `assistant`.
- The first message must be from `user`.
- The latest message must be from `user`.
- Roles must alternate.
- Message content must be non-empty.

### Success response schema

```json
{
  "reply": "string",
  "recommendations": [
    {
      "name": "string",
      "url": "https://www.shl.com/products/product-catalog/view/example/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

Rules:

- Response has exactly `reply`, `recommendations`, and `end_of_conversation`.
- Recommendation objects have exactly `name`, `url`, and `test_type`.
- `recommendations` has 0 to 10 items.
- Clarification and refusal turns return `recommendations: []`.
- Recommendation URLs must come from the scraped SHL catalog.

### Example recommendation request

```json
{
  "messages": [
    {
      "role": "user",
      "content": "We are hiring a senior backend engineer with Java, Spring, SQL, AWS, and Docker."
    }
  ]
}
```

### Example clarification response

```json
{
  "reply": "Should I prioritize backend, frontend, or balanced full-stack coverage for this technical role?",
  "recommendations": [],
  "end_of_conversation": false
}
```

### Example recommendation response

```json
{
  "reply": "I found SHL assessments for java, spring, sql. Top matches: Amazon Web Services (AWS) Development (New), Core Java (Advanced Level) (New), SQL (New). Retrieval confidence is 0.62.",
  "recommendations": [
    {
      "name": "Amazon Web Services (AWS) Development (New)",
      "url": "https://www.shl.com/products/product-catalog/view/amazon-web-services-aws-development-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

## Error responses

Errors never expose stack traces or secrets.

### Validation error

HTTP `422`

```json
{
  "error": {
    "code": "validation_error",
    "message": "Request schema is invalid.",
    "request_id": "..."
  }
}
```

### Catalog unavailable

HTTP `503`

```json
{
  "error": {
    "code": "catalog_error",
    "message": "Catalog is unavailable.",
    "request_id": "..."
  }
}
```

### Retrieval unavailable

HTTP `503`

```json
{
  "error": {
    "code": "retrieval_error",
    "message": "Retrieval index is unavailable.",
    "request_id": "..."
  }
}
```

### Model/provider unavailable

HTTP `503`

```json
{
  "error": {
    "code": "llm_error",
    "message": "Model provider is unavailable.",
    "request_id": "..."
  }
}
```

### Internal error

HTTP `500`

```json
{
  "error": {
    "code": "internal_error",
    "message": "An internal error occurred.",
    "request_id": "..."
  }
}
```
