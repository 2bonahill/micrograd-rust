use micrograd_rust::engine::Value;

extern crate micrograd_rust;

fn main() {
    println!("Welcome to Micrograd-Rust");
    simple();
    // manual_grad_simple();
}

pub fn simple() {
    let a: Value = Value::new(2.0);
    let b: Value = Value::new(3.0);
    let c: Value = a * b;
    // dbg!(&c);
    c.set_grad(1.0);
    c._backward();
    dbg!("AFter backward");
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
/*
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
    let grad = (o2.data - o1.data) / delta;
    println!("Grad: {}", grad);
}
*/
