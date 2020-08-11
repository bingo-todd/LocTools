# -*- coding: UTF-8 -*-

import smtplib
import argparse
from email.mime.text import MIMEText
from email.header import Header


def send_email(args):

    if isinstance(args.message, list):
        message_context = ' '.join(args.message)
    else:
        message_context = args.message

    message = MIMEText(message_context, 'plain', 'utf-8')
    message['From'] = Header("hearing_workstation", 'utf-8')
    message['To'] = Header("Monitor", 'utf-8')
    message['Subject'] = Header(args.subject, 'utf-8')

    try:
        with smtplib.SMTP("localhost") as server:
                server.sendmail(args.sender_address, args.receiver_address,
                                message.as_string())
    except smtplib.SMTPException:
        print('fail')


def parse_arg():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--sender-address', dest='sender_address',
                        default='from@runoob.com', type=str,
                        help='email address of sender')
    # parser.add_argument('--password', dest='password', required=True,
    #                     help="passwrod of sender's emall")
    parser.add_argument('--receiver-address', dest='receiver_address',
                        default='bh_song@163.com', type=str,
                        help='email address of receiver')
    parser.add_argument('--subject', dest='subject', default='Exp', nargs='+',
                        type=str, help='subject of email')
    parser.add_argument('--message', dest='message', type=str, nargs='+',
                        required=True, help='message to send')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arg()
    send_email(args)
