from twilio.rest import Client

SID = 'AC46ddfa7df38755a5aed35bce242e6ce2'
AUTH_TOKEN = '909fadb81d36c8ce598b313c3f0450c4'

cl = Client(SID, AUTH_TOKEN)

cl.messages.create(body='Hello123', from_='+12542563147', to='+918800345712')