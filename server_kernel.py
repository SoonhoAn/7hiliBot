import logging
import random
import re
import time

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram import ChatAction
from functools import wraps

from model import load_model
from post_processing import get_response

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler = logging.FileHandler('./chat.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def start_command(update, context):
    context.chat_data['interactions'] = []
    update.message.reply_text("[Chat Started]. Type \"Exit\" to quit conversation.")


def self_decorator(self, func):
    def command_func(update, context, *args, **kwargs):
        return func(self, update, context, *args, **kwargs)
    return command_func


# Cf) : https://github.com/python-telegram-bot/python-telegram-bot/wiki/Code-snippets#send-a-chat-action
def send_action(action):
    """Sends `action` while processing func command."""
    def decorator(func):
        @wraps(func)
        def command_func(self, update, context, *args, **kwargs):
            context.bot.send_chat_action(chat_id=update.effective_message.chat_id, action=action)
            return func(self, update, context, *args, **kwargs)
        return command_func
    return decorator


# Cf) : https://github.com/python-telegram-bot/python-telegram-bot/wiki/Code-snippets#send-a-chat-action
send_typing_action = send_action(ChatAction.TYPING)


# Cf) : https://github.com/microsoft/DialoGPT/blob/master/reddit_extractor/src/reddit.py
def gpt_normalize(txt):
    txt = re.sub(r"[^A-Za-z0-9()\[\]:,.!?'“”\"]", " ", txt)
    return ' '.join(txt.strip().split())


@send_typing_action
def message(self, update, context):
    memory = 1

    if 'interactions' not in context.chat_data:
        context.chat_data['interactions'] = []

    interactions = context.chat_data['interactions']

    input_msg = update.message.text
    if input_msg.lower() == 'exit':
        context.chat_data['interactions'] = []
        update.message.reply_text("See you again!")
        return None

    if memory == 0:
        context.chat_data['interactions'] = []

    interaction = {
        'input_msgs': [],
        'output_candidates': []
    }

    interactions.append(interaction)
    interaction['input_msgs'].append(input_msg)
    logger.info(f"ChatID:{update.effective_message.chat_id} - User : {input_msg}")

    start_time = time.time()
    dialogue = ""
    position = max(len(interactions)-memory-1, 0) if memory >= 0 else 0
    for interaction in interactions[position:]:
        for script in interaction['input_msgs']:
            dialogue += gpt_normalize(script) + self.tokenizer.eos_token
        for script in interaction['output_candidates']:
            dialogue += gpt_normalize(script) + self.tokenizer.eos_token

    output_candidates = get_response(self.model, self.tokenizer, dialogue)
    output_msg = random.choice(output_candidates)
    interaction['output_candidates'].append(output_msg)
    update.message.reply_text(output_msg)
    response_time = time.time() - start_time
    logger.info(f"ChatID:{update.effective_message.chat_id} - Bot : {output_msg} ({response_time:4f}secs)")


def error(update, context):
    logger.warning(context.error)


class ServerKernel:
    def __init__(self, model, tokenizer):
        logger.info("Initializing ChiliBot...")

        self.model = model
        self.tokenizer = tokenizer
        token = TOKEN

        self.updater = Updater(token, use_context=True)
        dp = self.updater.dispatcher
        dp.add_handler(MessageHandler(Filters.text, self_decorator(self, message)))
        dp.add_handler(CommandHandler('start', start_command))
        dp.add_error_handler(error)

    def run_chat(self):
        logger.info("Running ChiliBot...")
        self.updater.start_polling()
        self.updater.idle()


def main():
    model, tokenizer = load_model()
    server = ServerKernel(model, tokenizer)
    server.run_chat()


if __name__ == '__main__':
    main()
