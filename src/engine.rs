use std::{
    cell::RefCell,
    fmt::Display,
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
};

#[derive(Debug, PartialEq)]
pub enum Op {
    None,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Neg,
}

#[derive(Debug)]
pub struct Value {
    pub data: f64,
    pub grad: f64,
    pub _prev: Vec<Rc<RefCell<Value>>>,
    pub _op: Op,
}

impl Value {
    pub fn new(data: f64) -> Self {
        Value {
            data,
            grad: 0.0,
            _prev: Vec::new(),
            _op: Op::None,
        }
    }

    pub fn new_with_op(data: f64, op: Op) -> Self {
        Value {
            data,
            grad: 0.0,
            _prev: Vec::new(),
            _op: op,
        }
    }

    pub fn _backward(&self) {
        let mut c1 = self._prev[0].borrow_mut();
        let mut c2 = self._prev[1].borrow_mut();
        match self._op {
            Op::Add | Op::Sub => {
                // The operation was c1 + c2
                c1.grad = self.grad;
                c2.grad = self.grad;
            }
            Op::Mul => todo!(),
            Op::Div => todo!(),
            Op::Neg => todo!(),
            Op::Pow => todo!(),
            Op::None => {}
        }
    }

    pub fn pow(&self, exponent: f64) -> Self {
        let new_data = self.data.powf(exponent);
        Value::new_with_op(new_data, Op::Pow)
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self, other: Self) -> Self::Output {
        let mut out = Value {
            data: self.data + other.data,
            grad: 0.0,
            _prev: Vec::new(),
            _op: Op::Add,
        };

        out._prev.push(Rc::new(RefCell::new(self)));
        out._prev.push(Rc::new(RefCell::new(other)));

        out
    }
}

impl Add<f64> for Value {
    type Output = Value;

    fn add(self, other: f64) -> Self::Output {
        Value::new_with_op(self.data + other, Op::Add)
    }
}

impl Sub for Value {
    type Output = Value;

    fn sub(self, other: Self) -> Self::Output {
        Value::new_with_op(self.data - other.data, Op::Sub)
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, other: Self) -> Self::Output {
        Value::new_with_op(self.data * other.data, Op::Mul)
    }
}

impl Neg for Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        Value::new_with_op(-self.data, Op::Neg)
    }
}

impl Div for Value {
    type Output = Value;

    fn div(self, other: Self) -> Self::Output {
        Value::new_with_op(self.data / other.data, Op::Div)
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value(data={})", self.data)
    }
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_add() {
        let a: Value = Value::new(2.0);
        let b: Value = Value::new(3.0);
        let c: Value = a + b;
        assert_eq!(c.data, 5.0);
        assert_eq!(c._op, Op::Add);
    }

    #[test]
    fn test_sub() {
        let a: Value = Value::new(2.0);
        let b: Value = Value::new(3.0);
        let c: Value = a - b;
        assert_eq!(c.data, -1.0);
        assert_eq!(c._op, Op::Sub);
    }

    #[test]
    fn test_mul() {
        let a: Value = Value::new(2.0);
        let b: Value = Value::new(3.0);
        let c: Value = a * b;
        assert_eq!(c.data, 6.0);
        assert_eq!(c._op, Op::Mul);
    }

    #[test]
    fn test_pow() {
        let a: Value = Value::new(3.0);
        let b: Value = a.pow(2.0);
        assert_eq!(b.data, 9.0);
        assert_eq!(b._op, Op::Pow);
    }

    #[test]
    fn test_div() {
        let a: Value = Value::new(3.0);
        let b: Value = Value::new(2.0);
        let c: Value = a / b;
        assert_eq!(c.data, 1.5);
        assert_eq!(c._op, Op::Div);
    }

    #[test]
    fn test_neg() {
        let a: Value = Value::new(3.0);
        let b = -a;
        assert_eq!(b.data, -3.0);
        assert_eq!(b._op, Op::Neg);
    }
}
