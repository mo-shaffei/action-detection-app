from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram import KeyboardButton, ReplyKeyboardMarkup, ReplyKeyboardRemove
import signal
import pymongo
from datetime import datetime

mongo_Client = pymongo.MongoClient('localhost', 27017)
db = mongo_Client.webapp
results_data = db.results
state = None
authorized = ['shaffei']
alert = [911138085, 967475125]
last_smoking = datetime.now()
alarm_interval = 1
updater = None


def alarm_handler(signum, stack):
    global last_smoking
    result = results_data.find({'action': 'Smoking'}).sort("start", -1).limit(1)[0]
    if result['start'] > last_smoking:
        message = 'Alert Smoking Detected!\n'
        message += f'Time: {result["start"].strftime("%Y-%m-%d, %H:%M:%S")}\n'
        message += f'Building: {result["building"]}\n'
        message += f'Area: {result["location"]}\n'
        message += f'Confidence: {result["confidence"]}%'
        for user in alert:
            updater.bot.send_message(user, message)
        last_smoking = result['start']
    signal.alarm(alarm_interval)


def initialize_alert():
    global last_smoking
    result = results_data.find({'action': 'Smoking'}).sort("start", -1).limit(1)[0]
    last_smoking = result['start']


def is_authorized(update, context):
    if update.message.from_user['username'] in authorized:
        return True
    context.bot.send_message(chat_id=update.effective_chat.id, text="Access Denied")
    return False


def start(update, context):
    global state
    if not is_authorized(update, context):
        return
    state = 'location'
    locations = ["Helmy", "Nano"]
    buttons = [[KeyboardButton(location) for location in locations[:3]], [KeyboardButton('Cancel')]]
    context.bot.send_message(chat_id=update.effective_chat.id,
                             text="Choose Building", reply_markup=ReplyKeyboardMarkup(buttons))


def display_locations(location, update, context):
    global state
    if location == 'Helmy':
        locations = ['Biomedical Lab F024', 'Biology Lab B004']
        buttons = [[KeyboardButton(location) for location in locations[:3]], [KeyboardButton('Cancel')]]
        state = 'Helmy'
    elif location == 'Nano':
        locations = ['Computer Lab S012A', 'Computer Lab S012B', 'Electronics Lab S013']
        buttons = [[KeyboardButton(location) for location in locations[:3]], [KeyboardButton('Cancel')]]
        state = 'Nano'
    else:
        buttons = [[]]

    context.bot.send_message(chat_id=update.effective_chat.id,
                             text="Choose Area", reply_markup=ReplyKeyboardMarkup(buttons))


def send_data(n, update, context):
    global state
    building, area = state.split(',')
    results = results_data.find({'building': building, 'location': area}).sort("start", -1).limit(n)
    results = list(results)
    for result in results:
        message = f'Time: {result["start"].strftime("%Y-%m-%d, %H:%M:%S")}\n'
        message += f'Action: {result["action"]}\n'
        message += f'Confidence: {result["confidence"]}%'
        context.bot.send_message(chat_id=update.effective_chat.id, text=message)


def messageHandler(update, context):
    global state
    if not is_authorized(update, context):
        return
    print(update.message.from_user['username'])
    text = update.message.text
    if text in ["Helmy", "Nano"]:
        display_locations(text, update, context)
    elif text in ['Computer Lab S012A', 'Computer Lab S012B', 'Electronics Lab S013', 'Biomedical Lab F024',
                  'Biology Lab B004']:
        context.bot.send_message(chat_id=update.effective_chat.id, text='How many records do you want?',
                                 reply_markup=ReplyKeyboardRemove())
        state = f'{state},{text}'
    elif text.isdigit():
        send_data(int(text), update, context)
    elif text == 'Cancel':
        context.bot.send_message(chat_id=update.effective_chat.id, text='Operation Cancelled',
                                 reply_markup=ReplyKeyboardRemove())


def main():
    global updater
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(1)
    initialize_alert()
    with open('telegram-bot-token.txt', 'r') as file:
        token = file.read().strip()
    updater = Updater(token, use_context=True)
    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))

    # on noncommand i.e message - echo the message on Telegram
    dp.add_handler(MessageHandler(Filters.text, messageHandler))

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == "__main__":
    main()
