from datetime import datetime, timezone, tzinfo
from typing import Dict

from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

SYSTEM_PROMPT_STR = """
You serve as an intelligent home assistant. You are providing insightful responses to the user's inquiries and execute given commands.

You have the capability to manage the operation of in-home IOT devices. You furthermore have given the current date and time in UTC, and the user's time zone. Thereby, assume the user's current date and time. You can perform a web search to redeem information on latest. STRICTLY ENSURE to call a function (especially web search) only if ABSOLUTLY INDISPENSABLE for the successfull fulfillment of the user query. Use your own knowledge as much as possible to call as LESS functions as possible. STRICT ADHERENCE to the exact formats and parameters defined for these functions is ESSENTIAL.

The User input is transcribed from audio. The user receives your text responses in audio format. Make sure to respond with plain text. Always keep your responses as short as possible. Avoid explanations unless absolutely necessary or explicitly demanded by the user. Only answer in complete sentences if suitable. Strictly ensure to output speakable text. Always give your answer in the language of the user query. Only output the answer to the question or a short confirmation if a command was given. If a user command is unfeasible, unclear, or illogical, explain it to the user.
"""


# ADDITIONAL_INFORMATION_TEMPLATE_STR = """
# Additional information:
# Current date and time:
# {current_datetime}

# IOT Device information:

# Current lighting status:
# {iot_current_lighting}
# """

ADDITIONAL_INFORMATION_TEMPLATE_STR = """
Additional information:
Current date and time:
{current_datetime}
"""

CURRENT_DATETIME_PROMPT_STR = (
    "The current date and UTC time in ISO 8601 format is {datetime}. The user is in time zone {timezone}."
)

current_datetime_prompt_template = PromptTemplate(
    template=CURRENT_DATETIME_PROMPT_STR, input_variables=["datetime", "timezone"]
)


def current_datetime_data() -> Dict[str, tzinfo]:
    now = datetime.now(timezone.utc)
    return {"datetime": now.isoformat(), "timezone": now.astimezone().tzinfo}


def iot_current_lighting_data():
    return """
    [
        {{
            "room": "LivingRoom",
            "brightnessPercentage": 80,
            "color": "warmwhite",
        }},
        {{"room": "Kitchen", "brightnessPercentage": 0, "color": "warmwhite"}},
        {{"room": "Hallway", "brightnessPercentage": 0, "color": "white"}},
        {{"room": "Bedroom", "brightnessPercentage": 0, "color": "yellow"}},
    ]
    """


current_datetime_prompt = current_datetime_prompt_template.format(**current_datetime_data())
additional_info_template = PromptTemplate(
    template=ADDITIONAL_INFORMATION_TEMPLATE_STR,
    input_variables=[
        "current_datetime",  # "iot_current_lighting"
    ],
)
additional_info = additional_info_template.format(
    current_datetime=current_datetime_prompt,  # iot_current_lighting=iot_current_lighting_data()
)

system_message = SystemMessage(content=SYSTEM_PROMPT_STR)
additional_info_message = SystemMessage(content=additional_info)
chat_history = MessagesPlaceholder("chat_history", optional=True)
human_message = HumanMessagePromptTemplate.from_template("{input}")
agent_scratchpad = MessagesPlaceholder(variable_name="agent_scratchpad")


chat_prompt = ChatPromptTemplate.from_messages([system_message, additional_info_message, chat_history, human_message])
agent_prompt = ChatPromptTemplate.from_messages(
    [system_message, additional_info_message, chat_history, human_message, agent_scratchpad]
)
