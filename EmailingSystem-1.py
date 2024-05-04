from nylas import APIClient
import datetime
import random
#these variables need to be updated using the nylas dashboard
CLIENT_ID = "cfzodavfpmvbrbowthsaaotil"
CLIENT_SECRET = "1ogyshjsu7k6rmrpmggioleqc"
ACCESS_TOKEN = "M5aETnQciZCD5H7eUq8LgkZcmFAyp8"
nylas = APIClient(
    CLIENT_ID,
    CLIENT_SECRET,
    ACCESS_TOKEN,
)
def email_user(email, name, message,image_path):
    today = datetime.date.today()

    # Textual month, day and year
    today_date = today.strftime("%B %d, %Y")


    funny_img1 = "fitness.jpg"
    funny_img2 = "fitness1.jpg"
    funny_images = [funny_img1, funny_img2]
    random_img = random.randint(0, 1)
    attachment = open(image_path, 'rb')
    file = nylas.files.create()
    file.filename = 'cap.jpg'
    file.stream = attachment
    file.save()
    attachment.close()
    
    draft = nylas.drafts.create()
    draft.subject = "Performance Summary"
    # Email message as a strigfied HTML
    greeting = "<p style=\"font-size:30px; text-align: center; color:Red;\"> <b>Amazing Workout " + name + "! </b> </p> <br>"
    performance_summary = "<p style=\"font-size:20px; text-align: center;\"><b>Here is your Performance Summary<b> for "
    message = today_date + ":<br><br>" + message

    draft.body = greeting + performance_summary + message
    draft.to = [{'name': name, 'email': email}]
    draft.attach(file)
    draft.send()


def main():
    message="This is a test message like a demo email"
    email_user("chakrapriya7@gmail.com", "Priya", message)


# if __name__ == "__main__":
    # main()

