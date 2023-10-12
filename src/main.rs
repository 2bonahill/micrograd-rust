use micrograd_rust::engine::Value;

extern crate micrograd_rust;

fn main() {
    lecture_example();
}

pub fn lecture_example() {
    // as in Andrejs video
    let x1 = Value::new(2.0);
    let x2 = Value::new(0.0);

    let w1 = Value::new(-3.0);
    let w2 = Value::new(1.0);

    let b = Value::new(6.8813735870195432);

    let x1w1 = x1.clone() * w1.clone();
    let x2w2 = x2.clone() * w2.clone();

    let x1w1x2w2 = x1w1.clone() + x2w2.clone();

    let n = x1w1x2w2.clone() + b;
    let o = n.tanh();

    o.backward();

    // println!("x1 grad: {}", x1.grad());
    // println!("x2 grad: {}", x2.grad());
    // println!("w1 grad: {}", w1.grad());
    // println!("w2 grad: {}", w2.grad());
    // println!("x1w1 grad: {}", x1w1.grad());
    // println!("x2w2 grad: {}", x2w2.grad());
    // println!("x1w1x2w2 grad: {}", x1w1x2w2.grad());
    // println!("o data: {}", o.data());

    assert!(ae(n.data(), 0.8814), "Incorrect n");
    assert!(ae(o.data(), 0.7071), "Incorrect final output o");
    assert!(ae(x1.grad(), -1.5), "Incorrect x1.grad()");
    assert!(ae(x2.grad(), 0.5), "Incorrect x2.grad()");
    assert!(ae(w1.grad(), 1.0), "Incorrect w1.grad()");
    assert!(ae(w2.grad(), 0.0), "Incorrect w2.grad()");
    assert!(ae(x1w1.grad(), 0.5), "Incorrect x1w1.grad()");
    assert!(ae(x2w2.grad(), 0.5), "Incorrect x2w2.grad()");
    assert!(ae(x1w1x2w2.grad(), 0.5), "Incorrect x1w1x2w2.grad()");
    assert!(ae(n.grad(), 0.5), "Incorrect n.grad()");
}

// check if a and b are "almost" equal (taking floating point errors into account)
pub fn ae(a: f64, b: f64) -> bool {
    let eps = 0.0001;
    (a - b).abs() < eps
}
