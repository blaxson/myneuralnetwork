from graphics import *

def main():
    temp = [784, 16, 16, 10, 5, 5, 5]
    wth = 160 + (280 * (len(temp)-1))
    wth = wth if wth < 1280 else 1280
    win = GraphWin(title="my window", width=wth, height=900)
    win.setBackground("white")
    ln = Line(Point(0,800), Point(wth, 800))
    ln.draw(win)
    set_network(win, temp)
    win.getMouse()
    win.close()

def draw_large_layer(win, x_val):
    pt = Point(x_val, 400)
    pt.draw(win)
    pt = Point(x_val, 390)
    pt.draw(win)
    pt = Point(x_val,410)
    pt.draw(win)
    for i in range (0, 8):
        pt = Point(x_val, 40 +(i*45))
        cir = Circle(pt, 17.5)
        cir.draw(win)
    for i in range (0, 8):
        pt = Point(x_val, 760 -(i*45))
        cir = Circle(pt, 17.5)
        cir.draw(win)

def draw_layer(win, x_val, num_neurons):
    if num_neurons % 2:
        pt = Point(x_val, 400)
        cir = Circle(pt, 17.5)
        cir.draw(win)
        for i in range (0, (num_neurons // 2)):
            pt = Point(x_val, 355 - (i*45))
            cir = Circle(pt, 17.5)
            cir.draw(win)
        for i in range(0, (num_neurons // 2)):
            pt = Point(x_val, 445 + (i*45))
            cir = Circle(pt, 17.5)
            cir.draw(win)
    else:
        for i in range(0, (num_neurons // 2)):
            pt = Point(x_val, 377.5 - (i*45))
            cir = Circle(pt, 17.5)
            cir.draw(win)
        for i in range(0, (num_neurons // 2)):
            pt = Point(x_val, 422.5 + (i*45))
            cir = Circle(pt, 17.5)
            cir.draw(win)
    
    return None

def set_layer(win, x_val, num_neurons):
    if num_neurons > 16:
        draw_large_layer(win, x_val)
    else:
       draw_layer(win, x_val, num_neurons)
    return None

def set_network(win, nn_shape):
    for i in range(0, len(nn_shape)):
        set_layer(win, 80 + (i * 280), nn_shape[i])
    return None

main()