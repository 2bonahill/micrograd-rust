use micrograd_rust::engine::Value;

extern crate micrograd_rust;

fn main() {
    println!("Welcome to Micrograd-Rust");
    let a: Value = Value::new(2.0);
    let b: Value = Value::new(3.0);
    let mut c: Value = a + b;

    c.grad = 1.0;
    dbg!(&c);
    c._backward();
    dbg!(&c);
}

pub fn manual_grad() {
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
