import requests
from requests.auth import HTTPBasicAuth
from twilio.rest import Client

# --------------------- Twilio Credentials ---------------------
account_sid = "ACd9b398d97f100c83a34f6c173b0f05e6"
auth_token = "88f8091f7b73d9772737ff9e560d8d9a"
twilio_sms_from = "+17632735991"
twilio_whatsapp_from = "whatsapp:+14155238886"

# --------------------- Recipients ---------------------
# SMS Recipients
sms_recipients = [
   
    "+918838083432",
]

# WhatsApp Recipients (must be verified in Twilio)
whatsapp_recipients = [
    "whatsapp:+918778428081",
    "whatsapp:+918838083432",
    "whatsapp:+919150441890"
]

# --------------------- Alert Message ---------------------
city = "TAMILNADU State"
disaster = "Flood"
safety_alert = "⚠️ Take shelter immediately. Avoid travel."

alert_message = f"""
📍 Location: {city}
🌀 Disaster: {disaster}
⚠️ Alert: {safety_alert}
Stay safe. 🚨
"""

# --------------------- Send SMS ---------------------
def send_bulk_sms():
    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
    body_text = f"🚨 ALERT: {disaster} predicted in {city}. {safety_alert}"

    for number in sms_recipients:
        data = {
            "To": number,
            "From": twilio_sms_from,
            "Body": body_text
        }

        response = requests.post(url, data=data, auth=HTTPBasicAuth(account_sid, auth_token))

        if response.status_code == 201:
            print(f"✅ SMS sent to {number}")
        else:
            print(f"❌ Failed to send SMS to {number}: {response.text}")

# --------------------- Send WhatsApp ---------------------
def send_bulk_whatsapp():
    client = Client(account_sid, auth_token)

    for number in whatsapp_recipients:
        try:
            message = client.messages.create(
                body=alert_message,
                from_=twilio_whatsapp_from,
                to=number
            )
            print(f"✅ WhatsApp sent to {number}")
        except Exception as e:
            print(f"❌ Failed to send WhatsApp to {number}: {e}")

# --------------------- Call Functions ---------------------
send_bulk_sms()
send_bulk_whatsapp()
