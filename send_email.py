# from email import message
import os
import re
import imghdr
from email.message import EmailMessage
import ssl
import smtplib
from os import path
import io
import sys
import traceback
import inspect


# pathlib.Path(PATH4).mkdir(parents=True, exist_ok=True)
mail_regex = r'^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[\.]\w{2,8}[\.]{,1}\w{,2}'


def send_email(credentials_path='credentials.txt',
               subject='Avances en Servidor',
               reciever='sender',
               message=None, report=None,
               path2images=None,
               suffix4images=None, path2pdf=None):
    """Manda un correo a *reciever* con encabezado *subject* y contenido
        formado por *message* + *report* (documento)
        al cual anexa imagenes en la carpeta *path2images*. El envío se hace
        desde el correo registrado en *credentials_path*
    Arguments:
        credentials_path (str, optional):
            Ruta a documento con correo y contraseña (16 digitos)
            el formatro dento del texto es "{'mail':correo, 'ps':contraseña}".
            Defaults to 'credentials.txt'.
        subject (str, optional):
            Título de correo. Defaults to 'Avances en Servidor'.
        reciever (str, optional):
            Dirección de correo del receptor si esta se fija en 'sender' la
            el correo se envia a quien lo envío. Defaults to 'sender'.
        message (str, optional):
            Mensaje del correo. Defaults to None.
        report (str, optional):
            Admite una cadena que se apendiza al mensaje original o una ruta
            a un archivo de texto con información extra para apendizarla  al
            mensaje. Defaults to None.
        path2images (str, optional):
            Ruta al folder que contiene imagenes a apendizar en el correo.
            También admite una lista de rutas hacia las imagenes a apendizar
            Defaults to None.
        suffix4images (str, optional):
            Cadena con el sufijo de imagenes a adjuntar en el correo. Defaults
            to None.
    """

    if path.exists(credentials_path):
        with open(credentials_path) as file:
            dat = eval(file.read())
            logging_data = eval(dat)
            # breakpoint()
        sender = logging_data['mail']
        password = logging_data['ps']
        if reciever == 'sender':
            reciever = logging_data['mail']

        if message is None:
            message = ''' Listo!!'''
        if (type(report) == str) and (not path.isfile(report)):
            message = message + '\n\n REPORTE: \n\n' + report
        elif (type(report) == str) and (path.isfile(report)):
            with open(report) as rpt:
                message = message + '\n\n REPORTE: \n\n' + rpt.read()

        if suffix4images is None:
            suffix4images = ''

        em = EmailMessage()
        em['from'] = sender
        em['To'] = reciever
        em['Subject'] = subject
        em.set_content(message)
        context = ssl.create_default_context()
        if not (path2images is None):
            if path.isdir(path2images):
                fig_dict = {fig: path2images+'/' + fig for fig in os.listdir(
                    path2images) if suffix4images+'.png' in fig}
                for fig in fig_dict.keys():
                    with open(fig_dict[fig], 'rb') as png_fig:
                        content = png_fig.read()
                    em.add_attachment(content, maintype='image',
                                      subtype=imghdr.what(
                                          None, content), filename=fig)
            if type(path2images) == list:
                for fig in path2images:
                    if path.isfile(fig):
                        with open(fig, 'rb') as png_fig:
                            content = png_fig.read()
                        em.add_attachment(content, maintype='image',
                                          subtype=imghdr.what(
                                              None, content), filename=fig)

        if not (path2pdf is None):
            with open(path2pdf, 'rb') as content_file:
                content = content_file.read()
                em.add_attachment(content, maintype='application',
                                  subtype='pdf', filename='reporte.pdf')

        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(sender, password)
            # smtp.sendmail(sender,reciever,em.as_string())
            smtp.send_message(msg=em, from_addr=sender, to_addrs=reciever)
        print('mensaje enviado')
    else:
        print('el archivo de credenciales no existe')


def report_sender(func, args=[], reciever='sfernandezm97@ciencias.unam.mx'):
    """Evalua la funcion func y guarda registro de lo que se imprime en
        pantalla durante esta evaluación. func DEBE devolver dos cadenas,
        la primera (path_) indica la ruta donde se guardará un reporte con todo
        lo impreso en pantalla y la segunda (file_suffix) indica el sufijo a
        usar en el reporte generado. La ruta  del
        reporte es path_ + 'report' + file_suffix + '.txt'
    Args:
        func (function):
            función a correr y de la que se obtiene todo lo impreso en
            pantalla. Esta DEBE devolver dos cadenas, la primera (path_)
            indica la ruta donde se guardará un reporte con todo lo impreso
            en pantalla y que puede contener imagenes a anexar al correo.
            La segunda (file_suffix) indica el sufijo a usar en el reporte
            generado.
        args (list):
            Lista con los argumentos de la funcion a evaluar
            reciever (str, optional): correo del receptor del reporte.
            Defaults to 'lmm@ciencias.unam.mx'.
    """
    # Objeto de salida de impresiones de pantalla por default
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    try:
        print('-'*30 + ' PRINTED ON TERMINAL' + '-'*30 + '\n\n')

        # funcion a evaluar y de cual se obtiene el reporte
        if len(inspect.getfullargspec(func).args) != 0:
            path = func(*args)
            # file_suffix = path_.replace("/", "")
        else:
            path = ''
            func()
        # edición de reporte impreso
        # regresamos al objeto inical para impresion en pantalla
        sys.stdout = old_stdout
        printed_on_terminal = buffer.getvalue()

        # with open(path + 'output.txt', 'w') as file:
        #    file.write(printed_on_terminal)

        # Termina análisis y manda resultados
        path2pdf = path + \
            'reporte.pdf' if os.path.exists(path+'reporte.pdf') else None
        send_email(reciever=reciever, subject='Terminó entrenamiento '+path,
                   message=printed_on_terminal, path2pdf=path2pdf)

    except:
        type_error = str(sys.exc_info()[0]).split()[1].strip("'<>")
        # descripcion de error (dice "TypeError: ")
        description_error = sys.exc_info()[1]
        trace_back = sys.exc_info()[2]  # objeto error

        regex = r' line (?P<lin_num>[\d]+)'

        nuevo_reporte = []
        for cadena in traceback.format_tb(trace_back):
            corte_d_linea = cadena.split('\n')
            line_num = re.match(regex, corte_d_linea[0].split(',')[1])
            numero_d_linea = line_num.groups('line_num')[0]
            corte_d_linea[-2] = ' '*re.search(r'^[ ]+', corte_d_linea[-2]).span(
            )[1]+'---> (line '+numero_d_linea + ')\t'+corte_d_linea[-2].strip()
            nuevo_reporte.append('\n'.join(corte_d_linea))
        nuevo_reporte = ' '.join(nuevo_reporte)
        nuevo_reporte = 'Traceback :\n' + nuevo_reporte + \
            type_error + ': ' + str(description_error)
        send_email(reciever=reciever,
                   subject='Error en codigo del servidor',
                   message=nuevo_reporte)
