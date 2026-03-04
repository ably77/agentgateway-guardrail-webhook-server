import yaml

from pydantic import BaseModel, Field
from typing import List
from pydantic.json_schema import models_json_schema


### Guardrails API ###
# The following classes are used to define the body field in the request and response models for
# the Guardrails API.
class Message(BaseModel):
    """
    A single message in a conversation with an LLM.
    Each message has a role (who sent it) and content (what was sent).
    """

    role: str = Field(
        description="The role of the message sender in the conversation. Common values include:\n- 'system': System instructions or context\n- 'user': Messages from the end user\n- 'assistant': Responses from the AI assistant",
        examples=["system", "user", "assistant"],
    )

    content: str = Field(
        description="The actual text content of the message. Depending on the role, the content can be a question, instruction, or response.",
        examples=["You are a helpful AI assistant.", "What is the capital of France?", "The capital of France is Paris."],
    )


class PromptMessages(BaseModel):
    """
    A complete conversation or prompt to be sent to an LLM.
    The `messages` array represents the conversation history in chronological order.
    """

    messages: List[Message] = Field(
        default_factory=list,
        description="A sequence of messages that form a conversation or prompt. The order of messages matters, because it represents the conversation history.",
        examples=[
            [
                {"role": "system", "content": "You are a helpful AI assistant that provides concise answers."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 is 4."},
                {"role": "user", "content": "What about 3+3?"},
            ]
        ],
    )


class ResponseChoice(BaseModel):
    """
    Represents a single possible response from the LLM.
    Some LLMs might provide multiple alternative responses.
    """

    message: Message = Field(
        description="The AI assistant's response to the user's prompt. The `role` is typically 'assistant' and the `content` has the response text.",
        examples=[{"role": "assistant", "content": "The sum of 2 and 2 is 4."}],
    )


class ResponseChoices(BaseModel):
    """
    Contains all possible responses from the LLM for a given prompt.
    The `choices` array might contain one or more alternative responses.
    """

    choices: List[ResponseChoice] = Field(
        default_factory=list,
        description="A list of possible responses from the LLM. Some models might provide multiple alternative responses.",
        examples=[
            [
                {"message": {"role": "assistant", "content": "The sum of 2 and 2 is 4."}},
                {"message": {"role": "assistant", "content": "When you add 2 to 2, you get 4."}},
            ]
        ],
    )


# The following classes are used to define the request and response models for the Guardrails API.


class GuardrailsPromptRequest(BaseModel):
    """
    GuardrailsPromptRequest is the request model for the Guardrails prompt API.
    """

    body: PromptMessages = Field(
        description="The body object is a list of the Message JSON objects from the prompts in the request."
    )


class MaskAction(BaseModel):
    """
    The response model for the Mask action, which indicates the message has been modified.
    This can be used in GuardrailsPromptResponse or GuardrailsResponseResponse when responding to a GuardrailsPromptRequest or a GuardrailsResponseRequest respectively.
    """

    body: PromptMessages | ResponseChoices = Field(
        description="The body has the modified messages that masked out some of the original content. When used in a GuardrailPromptResponse, this should be PromptMessages. When used in GuardrailResponseResponse, this should be ResponseChoices."
    )

    reason: str | None = Field(
        description="The reason is a human readable string that explains the reason for the action.",
        default=None,
    )


class RejectAction(BaseModel):
    """
    The response model for the Reject action, which indicates the request should be rejected.
    This action causes a HTTP error response to be sent back to the end user.
    """

    body: str = Field(
        description="The rejection message that to be used for the HTTP error response body."
    )

    status_code: int = Field(
        description="The HTTP status code to be returned in the HTTP error response."
    )

    reason: str | None = Field(
        description="The reason is a human readable string that explains the reason for the action.",
        default=None,
    )


class PassAction(BaseModel):
    """
    The response model for the Pass action, which indicates no modification is done to the messages.
    """

    reason: str | None = Field(
        description="The reason is a human readable string that explains the reason for the action.",
        default=None,
    )


class GuardrailsPromptResponse(BaseModel):
    """
    GuardrailsPromptResponse is the response model for the Guardrails prompt API.
    """

    action: PassAction | MaskAction | RejectAction = Field(
        description="""
        The action to be taken based on the request. The following actions are available:

        1. PassAction:
           - No action is required
           - The request proceeds unchanged to the LLM

        2. MaskAction:
           - Some content in the request needs to be masked
           - The modified request is returned in the body field
           - The number of messages must match the original request

        3. RejectAction:
           - The request should be rejected
           - A specific HTTP status code and message are returned
           - The request will not be sent to the LLM
        """
    )


class GuardrailsResponseRequest(BaseModel):
    """
    GuardrailsResponseRequest is the request model for the Guardrails response API.
    """

    body: ResponseChoices = Field(
        description="The body object is a list of the Choice JSON objects that have the response content from the LLM."
    )


class GuardrailsResponseResponse(BaseModel):
    """
    GuardrailsResponseResponse is the response model for the Guardrails response API.
    """

    action: PassAction | MaskAction = Field(
        description="""
        The action to be taken on the response. The following actions are available:

        1. PassAction:
           - No action is required
           - The response proceeds unchanged to the user

        2. MaskAction:
           - Some content in the response needs to be masked
           - The modified response is returned in the body field
           - The number of choices must match the original response
        """
    )


def print_json_schema(models):
    _, schemas = models_json_schema(
        [(model, "validation") for model in models],
        ref_template="#/components/schemas/{model}",
    )
    openapi_schema = {
        "openapi": "3.1.0",
        "info": {
            "title": "AI Gateway Guardrails Webhook API",
            "version": "0.0.1",
        },
        "components": {
            "schemas": schemas.get("$defs"),
        },
    }
    print(yaml.dump(openapi_schema, sort_keys=False))


if __name__ == "__main__":
    print_json_schema(
        [
            Message,
            PromptMessages,
            ResponseChoice,
            ResponseChoices,
            MaskAction,
            RejectAction,
            PassAction,
            GuardrailsPromptRequest,
            GuardrailsPromptResponse,
            GuardrailsResponseRequest,
            GuardrailsResponseResponse,
        ]
    )
