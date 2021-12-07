from flask import Flask, request, abort
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import *
import json

#FlexMessage = json.load(open('BasicTable.json','r',encoding='utf-8'))

#print(type(FlexMessage))

app = Flask(__name__)
# LINE BOT info
line_bot_api = LineBotApi('YOUR_CHANNEL_ACCESS_TOKEN')
handler = WebhookHandler('YOUR_CHANNEL_SECRET')

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    print(body)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'


contents= json.load(open('BasicTable.json','r',encoding='utf-8'))
print(contents)

flex_message = FlexSendMessage(
    alt_text='hello',
    contents= json.load(open('BasicTable.json','r',encoding='utf-8'))
)



# Message event
@handler.add(MessageEvent)
def handle_message(event):
    message_type = event.message.type
    user_id = event.source.user_id
    reply_token = event.reply_token
    message = str(event.message.text)
    if(message == 'test1'):
        FlexMessage = json.load(open('BasicTable.json','r',encoding='utf-8'))
        line_bot_api.reply_message(reply_token, FlexSendMessage('test1',FlexMessage))
    elif(message == 'test2'):
        FlexMessage = json.load(open('returnPhotoTable.json','r',encoding='utf-8'))
        line_bot_api.reply_message(reply_token, FlexSendMessage('test2',FlexMessage))
    else:
        line_bot_api.reply_message(reply_token, TextSendMessage(text=message))
import os
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 80))
    app.run(host='0.0.0.0', port=port)
