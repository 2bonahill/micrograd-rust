use micrograd_rust::{engine::Value, nn::MLP};
use rand::Rng;

extern crate micrograd_rust;

fn main() {
    // engine();
    // nn_lecture();
    nn_is_bigger_than()
}

pub fn nn_is_bigger_than() {
    let mlp: MLP = MLP::new(3, vec![4, 4, 1]);

    let mut rng = rand::thread_rng();

    let mut xs: Vec<Vec<f64>> = vec![];
    let mut ys: Vec<Value> = vec![];

    for _ in 0..40 {
        let r1 = rng.gen_range(0.0..=10.0);
        let r2 = rng.gen_range(0.0..=10.0);
        let r3 = rng.gen_range(0.0..=10.0);
        xs.push(vec![r1, r2, r3]);
        let t: f64 = if r1 + r2 + r2 > 15.0 { 1.0 } else { -1.0 };
        ys.push(Value::new(t));
    }

    for _ in 0..100 {
        let y_pred: Vec<Value> = xs.iter().map(|x| mlp.call(x.clone())).collect();

        let mut loss: Value = Value::new(0.0);

        // loss.set_grad(1.0);
        for i in 0..ys.len() {
            let d = y_pred[i].clone() - ys[i].clone();
            let d_sq = d.pow(2.0);
            loss = loss + d_sq;
        }

        dbg!(&loss.data());

        for p in mlp.parameters() {
            p.borrow_mut().grad = 0.0;
        }

        loss.backward();

        for p in mlp.parameters() {
            let grad = p.borrow().grad;
            p.borrow_mut().data -= 0.0001 * grad;
        }
    }

    // make a prediction:
    let pred = mlp.call(vec![1.0, 1.0, 1.0]);
    println!("Pred sum = 3: {}", pred);
    let pred = mlp.call(vec![10.0, 1.0, 1.0]);
    println!("Pred sum = 12: {}", pred);
    let pred = mlp.call(vec![10.0, 10.0, 10.0]);
    println!("Pred sum = 30: {}", pred);
}

pub fn nn_lecture() {
    // Create the Network with 2 hidden layers (4 neurons each)
    let mlp: MLP = MLP::new(3, vec![4, 4, 1]);

    // The test data
    let xs: Vec<Vec<f64>> = vec![
        vec![2.0, 3.0, -1.0],
        vec![3.0, -1.0, 0.5],
        vec![0.5, 1.0, 1.0],
        vec![1.0, 1.0, -1.0],
    ];

    // Desired targets
    let ys = vec![
        Value::new(1.0),
        Value::new(-1.0),
        Value::new(-1.0),
        Value::new(1.0),
    ];

    for _ in 0..100 {
        // forward pass
        let y_pred: Vec<Value> = xs.iter().map(|x| mlp.call(x.clone())).collect();

        let mut loss: Value = Value::new(0.0);

        for i in 0..ys.len() {
            let d = y_pred[i].clone() - ys[i].clone();
            let d_sq = d.pow(2.0);
            loss = loss + d_sq;
        }

        // reset the gradients
        for p in mlp.parameters() {
            p.borrow_mut().grad = 0.0;
        }

        // backpropagation and gradient adjustment
        loss.backward();
        for p in mlp.parameters() {
            let grad = p.borrow().grad;
            p.borrow_mut().data -= 0.01 * grad;
        }
    }
}

pub fn engine() {
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
