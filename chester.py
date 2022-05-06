# from telegram.ext import Updater
# from telegram.ext import CallbackContext

# updater = Updater(token='5374814963:AAEhinhU8MDoFICF2nZkfsU1TTjy6Ipy8Cw', use_context=True)

# dispatcher = updater.dispatcher

# def start(update: Update, context: CallbackContext):
#     context.bot.send_message(chat_id=update.effective_chat.id, text="I'm a Chester, Ashkan's assistant")


from telegram import Bot

bot = Bot('5374814963:AAEhinhU8MDoFICF2nZkfsU1TTjy6Ipy8Cw')


bot.send_message("@chester_van_Ash_bot", "The Execution of the tast on Peregrine was failed")


