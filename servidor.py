from wsgiref.simple_server import make_server
def aplicacion(env, start_response):
    headers = [('Content-Type','text/plain')]
    start_response('200 ok', headers)

    return ['ayuda'. encode( 'utf-8')]

server = make_server('localhost',8000,aplicacion)

server.serve_forever ( )