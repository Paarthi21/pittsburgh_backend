import imaplib

import email

from email import policy

import time

import os




mail = imaplib.IMAP4_SSL("imap.gmail.com",993)

mail.login("digiverzinvoiceclaim@gmail.com","wvkvofuvbewbhdku")

mail.select('Inbox')




def fetchMail():

    type, data = mail.search(None, 'UNSEEN')
    mail_ids = data[0]
    id_list = mail_ids.split()
    mailbox = []

    for msgnum in data[0][::-1].split():
        # fetch the message by its ID
        typ, msg_data = mail.fetch(msgnum, '(RFC822)')
        raw_msg = email.message_from_bytes(msg_data[0][1])
        mailObj = {"from":raw_msg["From"],"subject":raw_msg["Subject"],"contents":[]} 

        for part in raw_msg.walk():

            if part.get_content_maintype() == 'multipart':
                continue

            if part.get('Content-Disposition') is None:
                continue
            # extract the filename and content of the attachment
            filename = part.get_filename()
            content = part.get_payload(decode=True)
            # print(filename)
            
            if filename.endswith('.png') or filename.endswith(".jpeg") or filename.endswith(".jpg") and content is not None:

                mailObj["contents"].append({"filename":filename,"content":content})
            
            elif filename.endswith('.pdf') or filename.endswith(".docx") and content is not None:

                mailObj["contents"].append({"filename":filename,"content":content})

        mailbox.append(mailObj)

    return mailbox


print([i['from'] for i in fetchMail()])

# print(fetchMail())