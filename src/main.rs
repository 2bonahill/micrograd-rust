use micrograd_rust::engine::Value;

extern crate micrograd_rust;

fn main() {
    println!("Welcome to Micrograd-Rust");
    // simple();
    // manual_grad_simple();
    // manual_grad_complex();
    // linreg_two_features();
    linreg_one_feature();
}

/// y = w*x + b
pub fn linreg_one_feature() {
    // y = w*x + b
    let x = Value::new(2.0);
    let w = Value::new(3.0);
    let b = Value::new(6.0);

    let xw = x.clone() * w.clone();
    let o = xw.clone() + b.clone();

    println!("x*w: {}", xw);
    println!("o: {}", o);

    o.backward();

    println!("Grads after backward:");
    println!("Grad of 'o' wrt. 'x': {}", x.grad());
    println!("Grad of 'o' wrt. 'w': {}", w.grad());
    println!("Grad of 'o' wrt. 'b': {}", b.grad());

    // let's check whether this is true
    let delta = 0.001;
    let x = Value::new(2.0 + delta);
    let w = Value::new(3.0);
    let b = Value::new(6.0);

    let xw = x.clone() * w.clone();
    let o2 = xw.clone() + b.clone();

    let grad = (o2.data() - o.data()) / delta;
    println!("Manually computed grad of 'o' wrt. 'x': {}", grad);
}

pub fn linreg_two_features() {
    let x1 = Value::new(2.0);
    let x2 = Value::new(2.0);

    let w1 = Value::new(-3.0);
    let w2 = Value::new(1.0);

    let b = Value::new(6.0);

    let x1w1 = x1 * w1;
    let x2w2 = x2 * w2;

    let x1w1x2w2 = x1w1 + x2w2;

    let n = x1w1x2w2 + b;
    let o = n.tanh();

    dbg!(&o);

    o.backward();
}

pub fn simple() {
    let a: Value = Value::new(2.0);
    let b: Value = Value::new(3.0);
    let c: Value = a * b;
    c.set_grad(1.0);
    c._backward();
    c.backward();
    dbg!(&c);
}

pub fn manual_grad_simple() {
    // o = a + b
    let delta: f64 = 0.001;

    // Base
    let a: Value = Value::new(2.0);
    let b: Value = Value::new(3.0);
    let o1: Value = a + b;
    println!("o1: {}", o1);

    // Notch
    let a: Value = Value::new(2.0);
    let a: Value = a + delta;
    let b: Value = Value::new(3.0);
    let o2: Value = a + b;
    println!("o2: {}", o2);

    // Grad
    let grad = (o2.data() - o1.data()) / delta;
    println!("Grad of o with respect to a: {}", grad);
    dbg!(&o2);
}

pub fn manual_grad_complex() {
    // o = a*b + c
    let delta: f64 = 0.001;

    // Base
    let a: Value = Value::new(2.0);
    let b: Value = Value::new(3.0);
    let c: Value = Value::new(5.0);
    let ab: Value = a * b;
    let o1: Value = ab + c;
    println!("o1: {}", o1);

    // Notch
    let a: Value = Value::new(2.0);
    let a: Value = a + delta;
    let b: Value = Value::new(3.0);
    let c: Value = Value::new(5.0);
    let ab: Value = a * b;
    let o2: Value = ab + c;
    println!("o2: {}", o2);

    // Grad
    let grad = (o2.data() - o1.data()) / delta;
    println!("Grad: {}", grad);
}
