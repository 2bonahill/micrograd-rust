use std::{
    cell::RefCell,
    collections::HashSet,
    f64::consts::E,
    fmt::Display,
    hash::Hash,
    ops::{Add, Deref, Div, Mul, Neg, Sub},
    rc::Rc,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Value(Rc<RefCell<ValueInner>>);

impl Value {
    pub fn new(data: f64) -> Self {
        Value(Rc::new(RefCell::new(ValueInner::new(data))))
    }

    pub fn new_from_inner(inner: ValueInner) -> Self {
        Value(Rc::new(RefCell::new(inner)))
    }

    pub fn data(&self) -> f64 {
        self.borrow().data
    }

    pub fn grad(&self) -> f64 {
        self.borrow().grad
    }

    pub fn op(&self) -> Op {
        self.borrow().op
    }

    pub fn set_grad(&self, new_grad: f64) {
        self.borrow_mut().grad = new_grad;
    }

    pub fn adjust_grad(&self, delta: f64) {
        self.borrow_mut().grad += delta;
    }

    pub fn pow(&self, exponent: f64) -> Self {
        let out = ValueInner {
            data: self.borrow().data.powf(exponent),
            grad: 0.0,
            prev: vec![self.clone(), Value::new(exponent)],
            op: Op::Pow,
        };

        Value::new_from_inner(out)
    }

    pub fn tanh(&self) -> Self {
        let out = ValueInner {
            data: self.borrow().data.tanh(),
            grad: 0.0,
            prev: vec![self.clone()],
            op: Op::Tanh,
        };

        Value::new_from_inner(out)
    }

    pub fn backward(&self) {
        // Topological order all of the children in the graph
        let mut topo: Vec<Value> = Vec::new();
        let mut visited: HashSet<Value> = HashSet::new();
        self.build_topo(&mut topo, &mut visited);
        // dbg!(&topo);

        self.set_grad(1.0);
        for v in topo.iter().rev() {
            v._backward(); // Assuming _backward method exists for Value
        }
    }

    fn build_topo(&self, topo: &mut Vec<Value>, visited: &mut HashSet<Value>) {
        if !visited.contains(self) {
            visited.insert(self.clone());
            for child in &self.borrow().prev {
                // Assuming _prev is a field in Value
                child.build_topo(topo, visited);
            }
            topo.push(self.clone());
        }
    }

    pub fn _backward(&self) {
        let inner = self.borrow();
        let op = inner.op;
        let grad = inner.grad;
        let prev = inner.prev.clone();

        match op {
            Op::Add => {
                // The operation was a + b
                prev[0].adjust_grad(grad * (1.0));
                prev[1].adjust_grad(grad * (1.0));
            }
            Op::Sub => {
                // The operation was a - b
                prev[0].adjust_grad(grad * (1.0));
                prev[1].adjust_grad(grad * (-1.0));
            }
            Op::Mul => {
                // The operation was a * b
                let a = prev[0].data();
                let b = prev[1].data();
                prev[0].adjust_grad(grad * b);
                prev[1].adjust_grad(grad * a);
            }
            Op::Div => {
                // The operation was a / b = a * (1/b)
                let a = prev[0].data();
                let b = prev[1].data();
                prev[0].adjust_grad(grad * (1.0 / b));
                prev[1].adjust_grad(grad * (-1.0) * (a / b.powf(2.0)));
            }
            Op::Pow => {
                // The operation was a ^ b
                let a = prev[0].data();
                let b = prev[1].data();
                prev[0].adjust_grad(grad * (b * a.powf(b - 1.0)));
            }
            Op::Tanh => {
                let x = prev[0].data();
                let t: f64 = (E.powf(2.0 * x) - 1.0) / (E.powf(2.0 * x) + 1.0);
                prev[0].set_grad(prev[0].grad() + (1.0 - t.powf(2.0)) * grad);
            }
            Op::None => {}
        }
    }
}

impl Deref for Value {
    type Target = Rc<RefCell<ValueInner>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.borrow().hash(state);
    }
}

impl Add<Value> for Value {
    type Output = Value;

    fn add(self, other: Self) -> Self::Output {
        let out = ValueInner {
            data: self.borrow().data + other.borrow().data,
            grad: 0.0,
            prev: vec![self.clone(), other.clone()],
            op: Op::Add,
        };

        Value::new_from_inner(out)
    }
}

impl Add<f64> for Value {
    type Output = Value;

    fn add(self, other: f64) -> Self::Output {
        self.borrow_mut().data += other;
        self
    }
}

impl Sub<Value> for Value {
    type Output = Value;

    fn sub(self, other: Self) -> Self::Output {
        let out = ValueInner {
            data: self.borrow().data - other.borrow().data,
            grad: 0.0,
            prev: vec![self.clone(), other.clone()],
            op: Op::Sub,
        };

        Value::new_from_inner(out)
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, other: Self) -> Self::Output {
        let out = ValueInner {
            data: self.borrow().data * other.borrow().data,
            grad: 0.0,
            prev: vec![self.clone(), other.clone()],
            op: Op::Mul,
        };

        Value::new_from_inner(out)
    }
}

impl Neg for Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        self.borrow_mut().data *= -1.0;
        self
    }
}

impl Div for Value {
    type Output = Value;

    fn div(self, other: Self) -> Self::Output {
        let out = ValueInner {
            data: self.borrow().data / other.borrow().data,
            grad: 0.0,
            prev: vec![self.clone(), other.clone()],
            op: Op::Div,
        };

        Value::new_from_inner(out)
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value(data={})", self.borrow().data)
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Op {
    None,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Tanh,
}

#[derive(Debug, Clone)]
pub struct ValueInner {
    pub data: f64,
    pub grad: f64,
    pub prev: Vec<Value>,
    pub op: Op,
}

impl ValueInner {
    pub fn new(data: f64) -> Self {
        ValueInner {
            data,
            grad: 0.0,
            prev: Vec::new(),
            op: Op::None,
        }
    }

    pub fn new_with_op(data: f64, op: Op) -> Self {
        ValueInner {
            data,
            grad: 0.0,
            prev: Vec::new(),
            op,
        }
    }
}

impl Hash for ValueInner {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.to_bits().hash(state);
        self.grad.to_bits().hash(state);
        self.prev.hash(state);
    }
}

impl PartialEq for ValueInner {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
            && self.grad == other.grad
            && self.prev.len() == other.prev.len()
            && self.op == other.op
    }
}

impl Eq for ValueInner {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a: Value = Value::new(2.0);
        let b: Value = Value::new(3.0);
        let c: Value = a + b;
        assert_eq!(c.data(), 5.0);
        assert_eq!(c.op(), Op::Add);
    }

    #[test]
    fn test_sub() {
        let a: Value = Value::new(5.0);
        let b: Value = Value::new(3.0);
        let c: Value = a - b;
        assert_eq!(c.data(), 2.0);
        assert_eq!(c.op(), Op::Sub);
    }

    #[test]
    fn test_mul() {
        let a: Value = Value::new(2.0);
        let b: Value = Value::new(3.0);
        let c: Value = a * b;
        assert_eq!(c.data(), 6.0);
        assert_eq!(c.op(), Op::Mul);
    }

    #[test]
    fn test_div() {
        let a: Value = Value::new(6.0);
        let b: Value = Value::new(3.0);
        let c: Value = a / b;
        assert_eq!(c.data(), 2.0);
        assert_eq!(c.op(), Op::Div);
    }

    #[test]
    fn test_pow() {
        let a: Value = Value::new(6.0);
        let b = a.pow(2.0);
        assert_eq!(b.data(), 36.0);
        assert_eq!(b.op(), Op::Pow);
    }

    #[test]
    fn test_neg() {
        let a: Value = Value::new(6.0);
        let b = -a;
        assert_eq!(b.data(), -6.0);
    }
}
