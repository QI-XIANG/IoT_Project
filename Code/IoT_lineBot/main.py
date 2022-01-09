import picamera
from servoMotion import *
import os, shutil
from flask import Flask, request, abort
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import *
import json

from datetime import datetime
app = Flask(__name__)
# LINE BOT info
line_bot_api = LineBotApi('YOUR_CHANNEL_ACCESS_TOKEN')
handler = WebhookHandler('YOUR_CHANNEL_SECRET')
camera = picamera.PiCamera()

#object to control servo motor
servoControl = servoMotion(1,1)

folder = 'imagepreview'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

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
    if(message == 'right'):
        FlexMessage = json.load(open('BasicTable.json','r',encoding='utf-8'))
        line_bot_api.reply_message(reply_token, FlexSendMessage('test1',FlexMessage))
        servoControl.turnRight()
    elif(message == 'left'):
        FlexMessage = json.load(open('returnPhotoTable.json','r',encoding='utf-8'))
        line_bot_api.reply_message(reply_token, FlexSendMessage('test2',FlexMessage))
        servoControl.turnLeft()
    elif(message == 'reset'):
        FlexMessage = json.load(open('returnPhotoTable.json','r',encoding='utf-8'))
        line_bot_api.reply_message(reply_token, FlexSendMessage('test2',FlexMessage))
        servoControl.resetStatus()
    else:
        line_bot_api.reply_message(reply_token, TextSendMessage(text=message))

arr = [0]

#handle postbackdata
@handler.add(PostbackEvent)
def handle_postback(event):
    postback_data = event.postback.data
    user_id = event.source.user_id
    reply_token = event.reply_token
    if postback_data == "getCurrentPhoto" or postback_data == "takePhoto":
        DIR = 'imagepreview/fit'+str(len(arr))+'.jpg'
        camera.capture(DIR)
        print('camera take the picture')
        FlexMessage = json.load(open('returnPhotoTable.json','r',encoding='utf-8'))
        url = "https://b1e3-140-115-214-31.ngrok.io/fit"+str(len(arr))+'.jpg'
        print(url)
        FlexMessage["hero"]["url"] = url
        FlexMessage["body"]["contents"][1]["contents"][2]["contents"][1]["text"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line_bot_api.reply_message(reply_token, FlexSendMessage('即時照片查看',FlexMessage))
        arr.append(0)
    elif postback_data == "resetStatus":
        line_bot_api.reply_message(reply_token, TextSendMessage(text="重置狀態中..."))
        servoControl.resetStatus()
    elif postback_data == "backToFunctionTable":
        FlexMessage = json.load(open('BasicTable.json','r',encoding='utf-8'))
        line_bot_api.reply_message(reply_token, FlexSendMessage('綜合功能表',FlexMessage))
    elif postback_data == "changeAngle":
        FlexMessage = json.load(open('changeRotation.json','r',encoding='utf-8'))
        line_bot_api.reply_message(reply_token, FlexSendMessage('鏡頭角度調整',FlexMessage))
    elif postback_data == "turnLeft":
        servoControl.turnLeft()
        if servoControl.getLeft()==0:
            line_bot_api.reply_message(reply_token, TextSendMessage(text="不能再向左轉囉~"))
        FlexMessage = json.load(open('changeRotation.json','r',encoding='utf-8'))
        line_bot_api.reply_message(reply_token, FlexSendMessage('鏡頭角度調整',FlexMessage))
    elif postback_data == "turnRight":
        servoControl.turnRight()
        if servoControl.getRight()==0:
            line_bot_api.reply_message(reply_token, TextSendMessage(text="不能再向右轉囉~"))
        FlexMessage = json.load(open('changeRotation.json','r',encoding='utf-8'))
        line_bot_api.reply_message(reply_token, FlexSendMessage('鏡頭角度調整',FlexMessage))
    '''elif postback_data == "takePhoto":
        FlexMessage = json.load(open('returnPhotoTable.json','r',encoding='utf-8')) 
        FlexMessage["body"]["contents"][1]["contents"][2]["contents"][1]["text"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line_bot_api.reply_message(reply_token, FlexSendMessage('即時照片查看',FlexMessage))'''
    

import os
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 80))
    app.run(host='0.0.0.0', port=port)