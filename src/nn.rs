use rand::{distributions::Uniform, prelude::Distribution};

use crate::engine::Value;

#[derive(Debug)]
pub struct Neuron {
    pub w: Vec<Value>,
    pub b: Value,
}

impl Neuron {
    /// n: number of incoming links
    pub fn new(nin: i32) -> Self {
        let mut rng = rand::thread_rng();
        let generator = Uniform::from(-1.0f64..=1.0f64);
        let w: Vec<Value> = (0..nin)
            .map(|_| Value::new(generator.sample(&mut rng)))
            .collect();
        let b: Value = Value::new(generator.sample(&mut rng));

        Self { w, b }
    }

    /// w * x + b
    pub fn call(&self, x: Vec<Value>) -> Value {
        let mut sum: Value = Value::new(0.0);
        for i in 0..self.w.len() - 1 {
            sum = sum + self.w[i].clone() * x[i].clone();
        }
        sum = sum + self.b.clone();
        sum = sum.tanh();
        sum
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut p: Vec<Value> = self.w.clone();
        p.push(self.b.clone());
        p
    }
}

#[derive(Debug)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
    pub nin: i32,
    pub nout: i32,
}

impl Layer {
    /// nin: number of incoming links per neuron
    /// nout: number of neurons in the layer
    pub fn new(nin: i32, nout: i32) -> Self {
        let neurons: Vec<Neuron> = (0..nout).map(|_| Neuron::new(nin)).collect();
        Self { neurons, nin, nout }
    }

    pub fn call(&self, x: Vec<Value>) -> Vec<Value> {
        let mut outs: Vec<Value> = vec![];
        for i in 0..self.neurons.len() {
            outs.push(self.neurons[i].call(x.clone()));
        }
        outs
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut p: Vec<Value> = vec![];
        for i in &self.neurons {
            p.extend(i.parameters());
        }
        p
    }
}

#[derive(Debug)]
pub struct MLP {
    pub layers: Vec<Layer>,
}

impl MLP {
    /// nouts: number of neurons per layer
    /// nin: number of incoming links into the MLP (#features)
    pub fn new(nin: i32, nouts: Vec<i32>) -> Self {
        let mut dims: Vec<i32> = vec![nin];
        dims.extend(nouts);
        let mut layers: Vec<Layer> = vec![];
        for i in 0..dims.len() - 1 {
            let l = Layer::new(dims[i], dims[i + 1]);
            layers.push(l);
        }
        Self { layers }
    }

    pub fn call(&self, x: Vec<f64>) -> Value {
        let mut xv: Vec<Value> = x.iter().map(|f| Value::new(*f)).collect();
        for l in &self.layers {
            xv = l.call(xv.clone());
        }
        xv[0].clone()
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut p: Vec<Value> = vec![];
        for i in &self.layers {
            p.extend(i.parameters());
        }
        p
    }
}
