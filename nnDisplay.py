from graphics import *

def main():
    temp = [3, 4, 5, 5]
    wth = 160 + (280 * (len(temp)-1))
    wth = wth if wth < 1360 else 1360
    win = GraphWin(title="my window", width=wth, height=900)
    win.setBackground("white")
    ln = Line(Point(0,800), Point(wth, 800))
    ln.draw(win)
    draw_large_layer(win, 80)
    draw_layer(win, 360, 16)
    draw_layer(win, 640, 13)
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
    else:
        for i in range(0, (num_neurons // 2)):
            pt = Point(x_val, 377.5 - (i *45))
            cir = Circle(pt, 17.5)
            cir.draw(win)
        for i in range(0, (num_neurons // 2)):
            pt = Point(x_val, 422.5 + (i * 45))
            cir = Circle(pt, 17.5)
            cir.draw(win)
    
    return None

def draw_even_layer(win, x_val, num_neurons):
    return None

def draw_odd_layer(win, x_val, num_neurons):
    return None

def set_layer(win, x_val, num_neurons):
    if num_neurons > 16:
        draw_large_layer(win, x_val)
    else:
        if num_neurons % 2:
            draw_odd_layer(win, x_val, num_neurons)
        else:
            draw_even_layer(win, x_val, num_neurons)
    return None

def set_neurons(nn_shape):
    return None

main()