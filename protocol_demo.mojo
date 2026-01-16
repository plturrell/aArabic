from python import Python

trait Drawable:
    fn draw(self)

struct Circle:
    fn draw(self): print("Drawing Circle")

fn render(d: Drawable): 
    d.draw()

fn main():
    render(Circle())
