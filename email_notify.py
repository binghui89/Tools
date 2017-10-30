import smtplib, sys
from email.mime.text import MIMEText
mail_host="smtp.gmail.com:587"  # SMTP server
mail_user="reagan.fruit"    # Username
mail_postfix="gmail.com"  # 
  
def send_mail(to_list, sub, content, mail_pass):  
    # For help go to the following link:
    # https://www.digitalocean.com/community/questions/unable-to-send-mail-through-smtp-gmail-com
    me="Binghui Li"+"<"+mail_user+"@"+mail_postfix+">"  
    msg = MIMEText(content,_subtype='plain',_charset='gb2312')  
    msg['Subject'] = sub  
    msg['From'] = me  
    msg['To'] = ";".join(to_list)  
    try:  
        server = smtplib.SMTP(mail_host)  
        server.ehlo()
        server.starttls()
        server.login(mail_user,mail_pass)  
        server.sendmail(me, to_list, msg.as_string())  
        server.close()  
        return True  
    except Exception, e:  
        print str(e)  
        return False  

if __name__ == '__main__':  
    mailto_list = ['reagan.fruit@gmail.com']
    if len(sys.argv) > 1:
        mail_pass = sys.argv[1]
    else:
        mail_pass = None

    if mail_pass:
        if send_mail(
            mailto_list,
            "Model run completed.",
            "The model run is completed!",
            mail_pass
        ):
            print "Email sent successfully!"  
        else:  
            print "Email sent failed."
    else:
        print "No password!\nEmail sent failed."
