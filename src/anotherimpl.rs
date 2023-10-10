use std::cell::RefCell;
use std::ops::Add;
use std::rc::{Rc, Weak};

type ValueRef = Rc<RefCell<Value>>;
type WeakValueRef = Weak<RefCell<Value>>;

struct Value {
    value: f32,
    neighbors: Vec<ValueRef>,
    parents: Vec<WeakValueRef>,
}

impl Value {
    fn new(value: f32) -> ValueRef {
        Rc::new(RefCell::new(Value {
            value,
            neighbors: Vec::new(),
            parents: Vec::new(),
        }))
    }

    fn add_neighbor(&mut self, neighbor: &ValueRef) {
        neighbor
            .borrow_mut()
            .parents
            .push(Rc::downgrade(&Rc::clone(neighbor)));
        self.neighbors.push(Rc::clone(neighbor));
    }
}

impl Add for ValueRef {
    type Output = ValueRef;

    fn add(self, other: Self) -> ValueRef {
        let left_value = self.borrow().value;
        let right_value = other.borrow().value;
        Value::new(left_value + right_value)
    }
}

fn main() {
    let value1 = Value::new(1.0);
    let value2 = Value::new(2.5);
    let value3 = Value::new(0.5);

    value1.borrow_mut().add_neighbor(&value2);
    value1.borrow_mut().add_neighbor(&value3);

    let result_value = value1 + value2;
    println!(
        "Result of adding value1 and value2: {}",
        result_value.borrow().value
    );
}
