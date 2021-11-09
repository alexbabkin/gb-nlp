import logging
import torch
import dialogflow

from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from transformers import AutoModelForCausalLM, AutoTokenizer

TOKEN = '2010441242:AAENBE65g-x74hx1iE4TtSl3S-AqgM-yIes'


def get_length_param(text: str) -> str:
    tokens_count = len(tokenizer.encode(text))
    if tokens_count <= 15:
        len_param = '1'
    elif tokens_count <= 50:
        len_param = '2'
    elif tokens_count <= 256:
        len_param = '3'
    else:
        len_param = '-'
    return len_param


# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
def start(update: Update, context: CallbackContext):
    update.message.reply_text('Здрасьте')


def answer(update: Update, context: CallbackContext):
    global step, chat_history_ids
    input_user = update.message.text  #input("===> User:")

    # encode the new user input, add parameters and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(
        f'|0|{get_length_param(input_user)}|' + input_user +
        tokenizer.eos_token + '|1|1|',
        return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids],
                              dim=-1) if step > 0 else new_user_input_ids

    # generated a response
    chat_history_ids = model.generate(
        bot_input_ids,
        num_return_sequences=1,
        max_length=512,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.6,
        mask_token_id=tokenizer.mask_token_id,
        eos_token_id=tokenizer.eos_token_id,
        unk_token_id=tokenizer.unk_token_id,
        pad_token_id=tokenizer.pad_token_id,
        device='cpu',
    )

    update.message.text = f"babkin_bot: {tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)}"
    update.message.reply_text(update.message.text)

    step = step + 1 if step < 9 else 0


def main():
    updater = Updater(token=TOKEN)  # Токен API к Telegram
    dispatcher = updater.dispatcher

    # Enable logging
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO)

    logger = logging.getLogger()

    updater = Updater(TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(
        MessageHandler(Filters.text & ~Filters.command, answer))

    updater.start_polling()
    updater.idle()


tokenizer = AutoTokenizer.from_pretrained("./model")
model = AutoModelForCausalLM.from_pretrained("./model")
step = 0

if __name__ == "__main__":
    main()
