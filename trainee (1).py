import streamlit as st
import mediapipe as mp
import numpy as np
import math
import time
import cv2
from EmailingSystem import email_user
from graph import single_plot,double_plot
from calc import cals
flag=1
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# user position
stage = None
message=" "
# count of correct movement
counter = 0
l_angles=[]
r_angles=[]
frames=[]
frame_count=0



@st.cache_resource()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def get_pos(img, results):
    landmarks = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        landmarks.append([id, cx, cy])
    return landmarks

def drawOn(img, p1, p2, p3, angle,lmList):
    x1, y1 = lmList[p1][1:]
    x2, y2 = lmList[p2][1:]
    x3, y3 = lmList[p3][1:]

    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.line(img, (x3, y3), (x2, y2), (0, 255, 0), 1)
    cv2.circle(img, (x1, y1), 5, (255, 0, 0), cv2.FILLED)
    cv2.circle(img, (x2, y2), 5, (255, 0, 0), cv2.FILLED)
    cv2.circle(img, (x3, y3), 5, (255, 0, 0), cv2.FILLED)
    cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)




st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('AI Fitness Trainer\n using MediaPipe and OpenCV')
app_mode = st.sidebar.selectbox('', ['Training', 'About App','Demos and Tutorials'])
if app_mode == 'About App':
    st.title("AI FITNESS TRAINEE")
    st.markdown(
        'In this application we are using Mediapipe for detecting exercise gestures and opencv for webcam reading  and StreamLit for creating the Web Graphical User Interface (GUI)')
    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    text_html = '<h3 style="background-color: rgb(0, 128, 131); color: #fff; text-decoration:none; padding: 5px;">INSTRUCTIONS TO USE</h3>'
    st.markdown(text_html, unsafe_allow_html=True)
    st.markdown('step-1: Choose training mode to start your fitness today')
    st.markdown('step-2: If you are not aware of what you are intended to do,be comfortable to go our demos and tutorial section')
    st.markdown('step-3: Now,it is the time to start your exercise,please provide Name and email')
    st.markdown('step-4: Select exercise fom dropdown')
    st.markdown('step-4: Select how many time you want to repeat the exercise')
    st.markdown('step-5: Tick on the start')
    st.markdown('step-6: Now,our trainee can be able to capture your pose')
    st.markdown('step-7: start doing exercise')
    st.markdown('step-8: count will be displayed on screen how many times you repeated')
    st.markdown('step-9: Finish those repetitions you mentioned,our trainee will count for you')
    st.markdown('step-10: Check your given mail inbox to see your performance report')
    text_html = '<h3 style="background-color: rgb(0, 128, 131); color: #fff; text-decoration:none; padding: 5px;">BE AWARE...</h3>'
    st.markdown(text_html, unsafe_allow_html=True)
    st.markdown('Make sure that you have proper Lighting')
    st.markdown('Ensure that your camera is working properly and in correct angle to capture you')
elif app_mode == 'Demos and Tutorials':
	#HIGH KNESS
	st.title("HIGH KNEES")
	link_html = '<a href="https://youtu.be/oDdkytliOqE?si=Mferh8hYmi14Rh7m" style="background-color: rgb(0, 128, 131); color: #fff; text-decoration:none; padding: 5px;">Video tutorial &rarr;</a>'
	st.markdown(link_html, unsafe_allow_html=True)
	hk_video_url = "https://www.shutterstock.com/shutterstock/videos/1060403879/preview/stock-footage-athletic-woman-doing-high-knee-exercise-at-home-home-training-workout-home-fitness-there-is-some.webm"
	st.video(hk_video_url)
	
	#SQUATS
	st.title("SQUATS")
	link_html = '<a href="https://youtu.be/4KmY44Xsg2w?si=eh8-DmSgjZptopSs" style="background-color: rgb(0, 128, 131); color: #fff; text-decoration:none; padding: 5px;">Video tutorial &rarr;</a>'
	st.markdown(link_html, unsafe_allow_html=True)
	s_video_url = "https://www.shutterstock.com/shutterstock/videos/1102020407/preview/stock-footage-female-butt-workout-squats-at-home-athletic-asian-woman-squats-workout-in-living-room-female.webm"
	st.video(s_video_url)
	
	#SHOULDER PRESS
	st.title("SHOULDER PRESS")
	link_html = '<a href="https://youtu.be/xe19t2_6yis?si=SIpCkA04UFBoJfs_" style="background-color: rgb(0, 128, 131); color: #fff; text-decoration:none; padding: 5px;">Video tutorial &rarr;</a>'
	st.markdown(link_html, unsafe_allow_html=True)
	shoulder_video_url = "https://ak.picdn.net/shutterstock/videos/1104019445/preview/stock-footage-medium-shot-of-young-sportsman-performing-barbell-shoulder-press-in-gym-man-holds-barbell-on-the.mp4"
	st.video(shoulder_video_url)
	
	#LATERAL CURLS
	st.title("LATERAL CURLS")
	link_html = '<a href="https://youtu.be/PzsMitRdI_8?si=Xt_bteImzFf_vP7m" style="background-color: rgb(0, 128, 131); color: #fff; text-decoration:none; padding: 5px;">Video tutorial &rarr;</a>'
	st.markdown(link_html, unsafe_allow_html=True)
	lcurls_video_url = "https://www.shutterstock.com/shutterstock/videos/1012307186/preview/stock-footage-bodybuilder-performs-seated-side-lateral-raise-reflection-of-exercise-in-mirror.webm"
	st.video(lcurls_video_url)
	#CURLS
	st.title("CURLS")
	link_html = '<a href="https://youtu.be/sYV-ki-1blM?si=Oi9cMYUAKZnPggP_" style="background-color: rgb(0, 128, 131); color: #fff; text-decoration:none; padding: 5px;">Video tutorial &rarr;</a>'
	st.markdown(link_html, unsafe_allow_html=True)
	curls_video_url = "https://www.shutterstock.com/shutterstock/videos/1097465161/preview/stock-footage-athlete-exercising-with-dumbbells-in-indoor-gym-fit-man-doing-biceps-triceps-curls-workout-in-gym.webm"
	st.video(curls_video_url)
	
	
	
	
	#LEFT CURLS
	st.title("Left curls")
	link_html = '<a href="https://www.youtube.com/shorts/cHxRJdSVIkA?feature=share" style="background-color: rgb(0, 128, 131); color: #fff; text-decoration:none; padding: 5px;">Video tutorial &rarr;</a>'
	st.markdown(link_html, unsafe_allow_html=True)
	lecurls_video_url ="https://www.shutterstock.com/shutterstock/videos/1030360235/preview/stock-footage-fitness-sport-weightlifting-and-bodybuilding-concept-man-exercising-with-dumbbells-at-home.webm"
	st.video(lecurls_video_url)
	
	
	#PUSHUPS
	st.title("PUSH UPS")
	link_html = '<a href="https://youtu.be/_UBOxUl0Sl4" style="background-color: rgb(0, 128, 131); color: #fff; text-decoration:none; padding: 5px;">Video tutorial &rarr;</a>'
	st.markdown(link_html, unsafe_allow_html=True)
	pushups_video_url = "https://www.shutterstock.com/shutterstock/videos/1100041747/preview/stock-footage-calisthenics-outdoor-bodyweight-workout-doing-push-ups-in-sunny-day.webm"
	st.video(pushups_video_url)
	
	#PLANK
	st.title("PLANK")
	link_html = '<a href="https://youtu.be/pvIjsG5Svck?si=ry3kol3TxHF7hu3g" style="background-color: rgb(0, 128, 131); color: #fff; text-decoration:none; padding: 5px;">Video tutorial &rarr;</a>'
	st.markdown(link_html, unsafe_allow_html=True)
	p_video_url = "https://www.shutterstock.com/shutterstock/videos/1063989619/preview/stock-footage-fitness-indian-woman-doing-plank-exercise-workout-in-gym-indoors.webm"
	st.video(p_video_url)
	
	#SIT UPS
	st.title("SIT UPS")
	link_html = '<a href="https://youtu.be/1fbU_MkV7NE?si=bcHWl9xa7PkFSsAu" style="background-color: rgb(0, 128, 131); color: #fff; text-decoration:none; padding: 5px;">Video tutorial &rarr;</a>'
	st.markdown(link_html, unsafe_allow_html=True)
	situps_video_url = "https://cdn.coverr.co/videos/coverr-ab-workout-in-the-park-7546/1080p.mp4"
	st.video(situps_video_url)
	
	
	#KNEE BEND
	st.title("KNEE BEND")
	link_html = '<a href="https://youtu.be/W4LuS9rK0gU?si=A5dTYz_OHyQmvHHd" style="background-color: rgb(0, 128, 131); color: #fff; text-decoration:none; padding: 5px;">Video tutorial &rarr;</a>'
	st.markdown(link_html, unsafe_allow_html=True)
	kb_video_url ="https://www.shutterstock.com/shutterstock/videos/1061478490/preview/stock-footage-caucasian-woman-spending-time-at-home-in-living-room-exercising-with-dumbbells-doing-squats-in.webm"
	st.video(kb_video_url)
	
	#CRUNCHES
	st.title("CRUNCHES")
	link_html = '<a href="https://youtu.be/Xyd_fa5zoEU?si=uK846BgYpam0H4X3" style="background-color: rgb(0, 128, 131); color: #fff; text-decoration:none; padding: 5px;">Video tutorial &rarr;</a>'
	st.markdown(link_html, unsafe_allow_html=True)
	cru_video_url = "https://www.shutterstock.com/shutterstock/videos/1041475927/preview/stock-footage-long-haired-girl-doing-bicycle-crunches-exercise-working-out-at-empty-studio-in-front-the-windows.webm"
	st.video(cru_video_url)
	
	#HALF PLOUGH POSE
	st.title("HALF PLOUGH POSE")
	link_html = '<a href="https://youtu.be/uqKbLaXSfWE" style="background-color: rgb(0, 128, 131); color: #fff; text-decoration:none; padding: 5px;">Video tutorial &rarr;</a>'
	st.markdown(link_html, unsafe_allow_html=True)
	hp_video_url = "https://www.shutterstock.com/shutterstock/videos/1105807867/preview/stock-footage-video-of-woman-performing-alternate-leg-ardha-halasana-this-exercise-strengthen-the-muscles-of.webm"
	st.video(hp_video_url)
	
	
	
elif app_mode == 'Training':
    st.title('Select Exercise')
    st.set_option('deprecation.showfileUploaderEncoding', False)
    suggestion = " "
    left, center, right = st.columns(3)
    name = st.text_input("Name", "")
    email = st.text_input("Email address", "")
    option = st.selectbox(
        'Exercise',
        ("Select One", "High Knees","Squats","Shoulder Press","Lateral Curls", "Curls", "Left Curl", "Push Ups", "Plank","Sit Up",
         "Knee Bend", "Crunches","Half Plough Pose"))
    press_time = st.select_slider(
        'How many times you want to perform',
        options=['0', '2', '10', '15', '20', '25', '30'])

    cal_value=cals(option,int(press_time))
    message = "You have done " + option + " for " + press_time + " times and burnt "+str(cal_value)+" calories"
    run = st.checkbox("Start")
    if option == "Left Curl":
        count = 0
        direction = 0
        prev_per = 0
        e = 0

    if option == "Push Ups":
        dir = 0

    if option == "Knee Bend":
        relax_counter = 0
        bent_counter = 0
        bent_time = 0
        relax_time = 0


    if option == "Crunches":
        f=0

    path=0
    # path="test/left_curl_demo.mp4"
    # path = "test/letral.mp4"
    # path = "test/curls_demo.mp4"
    # path = "test/squats1.mp4"
    # path = "test/shoulderpress_demo.mp4"
    # path = "test/highkness.mp4"
    # path = "test/pushup2_demo.mp4"
    # path = "test/KneeBendVideo.mp4"
    # path = "test/situps.mp4"
    # path = "test/crunches 1.mp4"
    # path = "test/plough.mp4"
    # time.sleep(2)
    while run:
        stframe = st.empty()
        cap = cv2.VideoCapture(path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_input = int(cap.get(cv2.CAP_PROP_FPS))
        fps = 0
        i = 0

        ltitle, ctitle, rtitle = st.columns(3)

        with ltitle:
            st.markdown("**Frame Rate**")
            kpi1_text = st.markdown("0")

        with ctitle:
            st.markdown("**Count**")
            kpi2_text = st.markdown("0")

        with rtitle:
            st.markdown("**Target**")
            kpi3_text = st.markdown("0")

        st.markdown("**Suggestions**")
        sugg = st.empty()
        sugg.markdown("---")

        st.markdown("<hr/>", unsafe_allow_html=True)

        with mp_pose.Pose(static_image_mode=False,
                          model_complexity=1,
                          smooth_landmarks=True,
                          enable_segmentation=False,smooth_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            prevTime = 0

            while cap.isOpened():
                ret, frame = cap.read()
                frame_count+=1
                if not ret:
                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                result = pose.process(frame)
                try:
                    landmarks = get_pos(frame, result)

                    rshoulder = landmarks[12][1:]
                    relbow = landmarks[14][1:]
                    rwrist = landmarks[16][1:]

                    lshoulder = landmarks[11][1:]
                    lelbow = landmarks[13][1:]
                    lwrist = landmarks[15][1:]

                    rhip = landmarks[24][1:]
                    rknee = landmarks[26][1:]
                    rankle = landmarks[28][1:]

                    lhip = landmarks[23][1:]
                    lknee = landmarks[25][1:]
                    lankle = landmarks[27][1:]

                    # angle_right_elbow = round(calculate_angle(rshoulder, relbow, rwrist))
                    # angle_left_elbow = round(calculate_angle(lshoulder, lelbow, lwrist))
                    #
                    # angle_right_knee = round(calculate_angle(rhip, rknee, rankle))
                    # angle_left_knee = round(calculate_angle(lhip, lknee, lankle))
                    #
                    # angle_right_shoulder = round(calculate_angle(rwrist, rshoulder, rhip))
                    # angle_left_shoulder = round(calculate_angle(lwrist, lshoulder, lhip))

                    # cv2.putText(frame, str(angle_right_elbow), (relbow[0], relbow[1] - 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    # cv2.putText(frame, str(angle_left_elbow), (lelbow[0], lelbow[1] - 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    # cv2.putText(frame, str(angle_right_knee), (rknee[0], rknee[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    #             0.5, (255, 255, 255), 1)
                    # cv2.putText(frame, str(angle_left_knee), (lknee[0], lknee[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    #             (255, 255, 255), 1)
                    # cv2.putText(frame, str(angle_right_shoulder), (rshoulder[0] + 10, rshoulder[1] + 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    # cv2.putText(frame, str(angle_left_shoulder), (lshoulder[0] - 50, lshoulder[1] + 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    if option == "Left Curl":
                        angle_left_elbow = round(calculate_angle(lshoulder, lelbow, lwrist))
                        l_angles.append(angle_left_elbow)
                        frames.append(frame_count)
                        per = np.interp(angle_left_elbow, (45, 155), (100, 0))
                        bar = np.interp(angle_left_elbow, (45, 155), (60, 420))
                        drawOn(frame, 11, 13, 15, angle_left_elbow, landmarks)

                        color = (255, 0, 255)
                        if per == 100:
                            color = (0, 255, 0)
                            if direction == 0:
                                direction = 1
                                counter += 1
                            else:
                                suggestion="Lift Arm Down"

                        if per == 0:
                            if direction == 1:
                                direction = 0
                            else:
                                suggestion="Lift Arm Up"


                        cv2.putText(frame, f"stage: {str(suggestion)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                    cv2.LINE_AA)


                        cv2.rectangle(frame, (520, 60), (590, 420), color, 2)
                        cv2.rectangle(frame, (520, int(bar)), (590, 420), color, cv2.FILLED)
                        cv2.putText(frame, f'{int(per)} %', (520, 50), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)



                    if option == "Curls":
                        angle_right_elbow = round(calculate_angle(rshoulder, relbow, rwrist))
                        angle_left_elbow = round(calculate_angle(lshoulder, lelbow, lwrist))
                        drawOn(frame, 11, 13, 15, angle_left_elbow, landmarks)
                        l_angles.append(angle_left_elbow)
                        r_angles.append(angle_right_elbow)
                        frames.append(frame_count)
                        drawOn(frame, 12, 14, 16, angle_right_elbow, landmarks)
                        color = (255, 0, 255)
                        if angle_left_elbow >= 165 and angle_right_elbow >= 165:
                            stage = "open"
                            suggestion = "Keep elbows close."
                        elif angle_left_elbow <= 34 and angle_right_elbow <= 34 and stage == "open":
                            stage = "close"
                            counter += 1
                            suggestion = "Make sure you are lifting the weights fully"

                        # visualizing angle_left_elbow
                        per1 = np.interp(angle_left_elbow, (34, 165), (100, 0))
                        bar1 = np.interp(angle_left_elbow, (34, 165), (60, 420))
                        per2 = np.interp(angle_right_elbow, (34, 165), (100, 0))
                        bar2 = np.interp(angle_right_elbow, (34, 165), (60, 420))

                        if per1 == 100 and per2==100:
                            color=(0,255,0)

                        cv2.rectangle(frame, (520, 60), (590, 420), color, 2)
                        cv2.rectangle(frame, (520, int(bar1)), (590, 420), color, cv2.FILLED)
                        cv2.putText(frame, f'{int(per1)} %', (520, 50), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                        cv2.rectangle(frame, (20, 60), (90, 420), color, 2)
                        cv2.rectangle(frame, (20, int(bar2)), (90, 420), color, cv2.FILLED)
                        cv2.putText(frame, f'{int(per2)} %', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)


                        cv2.putText(frame, f"stage: {str(suggestion)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2, cv2.LINE_AA)

                    if option == "Lateral Curls":
                        angle_right_shoulder = round(calculate_angle(rwrist, rshoulder, rhip))
                        angle_left_shoulder = round(calculate_angle(lwrist, lshoulder, lhip))
                        l_angles.append(angle_left_shoulder)
                        r_angles.append(angle_right_shoulder)
                        frames.append(frame_count)
                        drawOn(frame, 15, 11, 23, angle_left_shoulder, landmarks)
                        drawOn(frame, 16, 12, 24, angle_right_shoulder, landmarks)
                        color = (255, 0, 255)
                        if angle_right_shoulder >= 100 and angle_left_shoulder >= 100 and stage =="down":
                            stage = "up"
                            counter += 1
                            suggestion = "Down"
                        elif 30 >= angle_right_shoulder and 30 >= angle_left_shoulder:
                            stage = "down"
                            suggestion = "Up"



                        # visualizing angle_left_shoulder
                        per1 = np.interp(angle_left_shoulder, (30, 100), (0, 100))
                        bar1 = np.interp(angle_left_shoulder, (30, 100), (420, 60))
                        per2 = np.interp(angle_right_shoulder, (30, 100), (0, 100))
                        bar2 = np.interp(angle_right_shoulder, (30, 100), (420, 60))

                        if per1 == 100 and per2==100:
                            color=(0,255,0)

                        cv2.rectangle(frame, (520, 60), (590, 420), color, 2)
                        cv2.rectangle(frame, (520, int(bar1)), (590, 420), color, cv2.FILLED)
                        cv2.putText(frame, f'{int(per1)} %', (520, 50), cv2.FONT_HERSHEY_PLAIN, 2, color,2)
                        cv2.rectangle(frame, (20, 60), (90, 420), color, 2)
                        cv2.rectangle(frame, (20, int(bar2)), (90, 420), color, cv2.FILLED)
                        cv2.putText(frame, f'{int(per2)} %', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)



                        cv2.putText(frame, f"stage: {str(suggestion)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2, cv2.LINE_AA)



                    if option == "Squats":
                        angle_right_knee = round(calculate_angle(rhip, rknee, rankle))
                        angle_left_knee = round(calculate_angle(lhip, lknee, lankle))
                        l_angles.append(angle_left_knee)
                        r_angles.append(angle_right_knee)
                        frames.append(frame_count)
                        drawOn(frame, 24, 26, 28, angle_right_knee, landmarks)
                        drawOn(frame, 23, 25, 27, angle_left_knee, landmarks)
                        color = (255, 0, 255)
                        if angle_right_knee >= 173 and angle_left_knee >= 173:
                            stage = "down"
                            suggestion = "Go Down"
                        elif angle_right_knee <= 90 and angle_left_knee <= 90 and stage == 'down':
                            stage = "up"
                            counter += 1
                            suggestion = "Go Up"
                        cv2.putText(frame, f"stage: {str(suggestion)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2, cv2.LINE_AA)

                        # visualizing angle_left_knee
                        per1 = np.interp(angle_left_knee, (90, 173), (100, 0))
                        bar1 = np.interp(angle_left_knee, (90, 173), (60, 420))


                        # visualizing angle_right_knee
                        per2 = np.interp(angle_right_knee, (90, 173), (100, 0))
                        bar2 = np.interp(angle_right_knee, (90, 173), (60, 420))

                        if per1 == 100 and per2==100:
                            color=(0,255,0)

                        cv2.rectangle(frame, (520, 60), (590, 420), color, 2)
                        cv2.rectangle(frame, (520, int(bar1)), (590, 420), color, cv2.FILLED)
                        cv2.putText(frame, f'{int(per1)} %', (520, 50), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                        cv2.rectangle(frame, (20, 60), (90, 420), color, 2)
                        cv2.rectangle(frame, (20, int(bar2)), (90, 420), color, cv2.FILLED)
                        cv2.putText(frame, f'{int(per2)} %', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

                    if option == "Push Ups":
                        angle_left_elbow = round(calculate_angle(lshoulder, lelbow, lwrist))
                        drawOn(frame, 11, 13, 15, angle_left_elbow, landmarks)
                        color = (255, 0, 255)
                        l_angles.append(angle_left_elbow)
                        frames.append(frame_count)
                        per = np.interp(angle_left_elbow, (70, 140), (100, 0))
                        bar = np.interp(angle_left_elbow, (70, 140), (60, 420))
                        if per == 100:
                            color = (0, 255, 0)
                            if dir == 0:
                                counter += 1
                                dir = 1
                            suggestion="Up"
                        if per == 0:
                            if dir == 1:
                                dir = 0
                            suggestion="Down"
                        cv2.putText(frame, f"stage: {str(suggestion)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.rectangle(frame, (520, 60), (590, 420), color, 2)
                        cv2.rectangle(frame, (520, int(bar)), (590, 420), color, cv2.FILLED)
                        cv2.putText(frame, f'{int(per)} %', (520, 50), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

                    if option == "Shoulder Press":
                        angle_right_elbow = round(calculate_angle(rshoulder, relbow, rwrist))
                        angle_left_elbow = round(calculate_angle(lshoulder, lelbow, lwrist))
                        l_angles.append(angle_left_elbow)
                        r_angles.append(angle_right_elbow)
                        frames.append(frame_count)
                        drawOn(frame, 11, 13, 15, angle_left_elbow, landmarks)
                        drawOn(frame, 12, 14, 16, angle_right_elbow, landmarks)
                        color=(255,0,255)
                        if angle_left_elbow > 170 and angle_right_elbow > 170:
                            stage = "Up"
                        if angle_left_elbow < 90 and angle_right_elbow < 90 and stage == 'Up':
                            stage = "Down"
                            counter += 1

                        cv2.putText(frame, f"stage: {str(suggestion)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2, cv2.LINE_AA)
                        # visualizing angle_left_elbow
                        per1 = np.interp(angle_left_elbow, (90, 170), (0, 100))
                        bar1 = np.interp(angle_left_elbow, (90, 170), (420, 60))
                        per2 = np.interp(angle_right_elbow, (90, 170), (0, 100))
                        bar2 = np.interp(angle_right_elbow, (90, 170), (420, 60))
                        if per1==100 and per2==100:
                            color=(0,255,0)
                        cv2.rectangle(frame, (520, 60), (590, 420), color, 2)
                        cv2.rectangle(frame, (520, int(bar1)), (590, 420), color, cv2.FILLED)
                        cv2.putText(frame, f'{int(per1)} %', (520, 50), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

                        # visua;izing angle_right_elbow

                        cv2.rectangle(frame, (20, 60), (90, 420), color, 2)
                        cv2.rectangle(frame, (20, int(bar2)), (90, 420), color, cv2.FILLED)
                        cv2.putText(frame, f'{int(per2)} %', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

                    if option == "High Knees":
                        angle_left_hip = round(calculate_angle(lshoulder, lhip, lknee))
                        drawOn(frame, 11, 23, 25, angle_left_hip, landmarks)
                        if angle_left_hip > 160:
                            stage = "Down"
                        if angle_left_hip < 80 and stage == 'Down':
                            stage = "Up"
                            counter += 1
                        cv2.putText(frame, f"stage: {str(suggestion)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2, cv2.LINE_AA)
                        per = np.interp(angle_left_hip, (80, 160), (0, 100))
                        bar = np.interp(angle_left_hip, (80, 160), (420, 60))
                        cv2.rectangle(frame, (520, 60), (590, 420), color, 2)
                        cv2.rectangle(frame, (520, int(bar)), (590, 420), color, cv2.FILLED)
                        cv2.putText(frame, f'{int(per)} %', (520, 50), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

                    if option == "Knee Bend":
                        angle_left_knee = round(calculate_angle(lhip, lknee, lankle))
                        drawOn(frame, 23, 25, 27, angle_left_knee, landmarks)
                        if angle_left_knee > 140:
                            relax_counter += 1
                            bent_counter = 0
                            stage = "Relaxed"
                            suggestion = ""

                        if angle_left_knee < 140:
                            relax_counter = 0
                            bent_counter += 1
                            stage = "Bent"
                            suggestion = ""

                            # rep
                        if bent_counter == 8:
                            counter += 1
                            suggestion = 'Rep completed. Relax knee'

                        elif bent_counter < 8 and stage == 'Bent':
                            suggestion = 'Keep Your Knee Bent'

                        else:
                            suggestion = " "

                        cv2.putText(frame, f"stage: {str(suggestion)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2, cv2.LINE_AA)

                    if option == "Sit Up":
                        shoulder_avg = [(rshoulder[0] + lshoulder[0]) / 2, (rshoulder[1] + lshoulder[1]) / 2]
                        hip_avg = [(rhip[0] + lhip[0]) / 2, (rhip[1] + lhip[1]) / 2]
                        knee_avg = [(rknee[0] + lknee[0]) / 2, (rknee[1] + lknee[1]) / 2]
                        angle = calculate_angle(shoulder_avg, hip_avg, knee_avg)
                        drawOn(frame, 11, 23, 25, angle_left_knee, landmarks)

                        if angle <= 84 and status=="Up":
                            counter += 1
                            status = "Down"
                            suggestion = "Down"

                        if angle >= 90:
                            status = "Up"
                            suggestion = "Up"

                        cv2.putText(frame, f"stage: {str(suggestion)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2, cv2.LINE_AA)
                        per = np.interp(angle, (84, 90), (0, 100))
                        bar = np.interp(angle, (84, 90), (420, 60))
                        cv2.rectangle(frame, (520, 60), (590, 420), color, 2)
                        cv2.rectangle(frame, (520, int(bar)), (590, 420), color, cv2.FILLED)
                        cv2.putText(frame, f'{int(per)} %', (520, 50), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

                    if option == "Crunches":
                        x1 = landmarks[0][1]
                        x2 = landmarks[12][1]

                        length = x1 - x2
                        print(length)
                        if length >= 0 and f == 0:
                            f = 1
                            stage = "Bend Forward"
                            suggestion=stage
                        elif length < 0 and f == 1:
                            f = 0
                            stage = "Relax"
                            suggestion=stage
                            counter += 1
                        drawOn(frame, 0, 12, 24, length, landmarks)

                        cv2.putText(frame, f"stage: {str(stage)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2, cv2.LINE_AA)


                    if option == "Plank":
                        angle1 = round(calculate_angle(lshoulder, lelbow, lwrist))
                        angle2 = round(calculate_angle(lshoulder, lhip, lknee))
                        angle3 = angle_left_knee = round(calculate_angle(lhip, lknee, lankle))
                        drawOn(frame, 11, 13, 15, angle1, landmarks)
                        drawOn(frame, 11, 23, 25, angle2, landmarks)
                        drawOn(frame, 23, 25, 27, angle3, landmarks)
                        if not 75 <= angle1 <= 105:
                            suggestion1='Bring your shoulder vertically above your elbow'
                            cv2.putText(frame, f"{str(suggestion1)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2, cv2.LINE_AA)
                        elif angle2 < 140:
                            suggestion2='Make your back straight. Bring your buttocks DOWN'
                            cv2.putText(frame, f"{str(suggestion2)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2, cv2.LINE_AA)
                        elif angle2 > 170:
                            suggestion3='Make your back straight. Bring your buttocks UP'
                            cv2.putText(frame, f"{str(suggestion3)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2, cv2.LINE_AA)
                        elif angle3 <= 160:
                            suggestion4='Do not bend your knee. Stretch your legs'
                            cv2.putText(frame, f"{str(suggestion4)}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2, cv2.LINE_AA)
                        else:
                            counter+=1
                            time.sleep(1)

                    if option == "Half Plough Pose":
                        # Calculate angles
                        angle_left_hip = round(calculate_angle(lshoulder, lhip, lknee))
                        drawOn(frame, 11, 23, 25, angle_left_hip, landmarks)

                        print(lhip[1], lknee[1])
                        # Determine if Half Plough Pose is being performed
                        if angle_left_hip<83:
                            suggestion1="Half Plough Pose detected!"
                            cv2.putText(frame, f"{str(suggestion1)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2, cv2.LINE_AA)
                            if lhip[1] > lknee[1]:
                                suggestion2="Your form looks good."
                                cv2.putText(frame, f"{str(suggestion2)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 0, 255), 2, cv2.LINE_AA)
                                counter += 1
                                time.sleep(1)
                            else:
                                suggestion3="Your hips should be below your knees for proper form."
                                cv2.putText(frame, f"{str(suggestion3)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 0, 255), 2, cv2.LINE_AA)
                        else:
                            suggestion4="Not Half Plough Pose."
                            cv2.putText(frame, f"{str(suggestion4)}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2, cv2.LINE_AA)


                    if counter == int(press_time):
                        counter = 0
                        press_time=0
                        suggestion = "Completed"+message
                        if option =="Left Curl":
                            single_plot(frames, l_angles, "Left Elbow", 45, 155)
                        if option =="Curls":
                            double_plot(frames, l_angles, r_angles, "Left Elbow", "Right Elbow", 34, 165)
                        if option =="Lateral Curls":
                            double_plot(frames, l_angles, r_angles, "Left Shoulder", "Right Shoulder", 30, 85)
                        if option =="Squats":
                            double_plot(frames, l_angles, r_angles, "Left Knee", "Right Knee", 90, 173)
                        if option =="Shoulder Press":
                            double_plot(frames, l_angles, r_angles, "Left Knee", "Right Knee", 90, 170)
                        if option =="Push Ups":
                            single_plot(frames, l_angles, "Left Elbow", 70, 140)
                        option = False
                        ret, frame = cap.read()
                        if ret:
                            # Display the captured image
                            st.image(frame, caption="Captured Image", use_column_width=True)
                            # Save the captured image
                            cv2.imwrite("cap.png", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        if flag:
                            email_user(email, name, message,"cap.png")
                            flag=0
                        run = False
                        frame_count = 0
                        frames = []
                        l_angles = []
                        r_angles = []
                        cap.release()

                except:
                    pass

                mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                currTime = time.time()
                fps = 1 / (currTime - prevTime)
                prevTime = currTime

                kpi1_text.write(f"<h1 style='text-align: left; color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
                kpi2_text.write(f"<h1 style='text-align: left; color: red;'>{counter}</h1>", unsafe_allow_html=True)
                kpi3_text.write(f"<h1 style='text-align: left; color: red;'>{press_time}</h1>", unsafe_allow_html=True)
                sugg.write(f"<h1 style='text-align: left; color: red;'>{suggestion}</h1>", unsafe_allow_html=True)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)
                frame = image_resize(image=frame, width=640)
                stframe.image(frame, channels='BGR', use_column_width=True)
