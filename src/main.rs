use tch::{nn, nn::Module, nn::OptimizerConfig, Tensor};

#[derive(Debug)]
struct XorNet {
    layer1: nn::Linear,
    layer2: nn::Linear,
}

impl XorNet {
    fn new(vs: &nn::Path) -> XorNet {
        let layer1 = nn::linear(vs, 2, 2, Default::default());
        let layer2 = nn::linear(vs, 2, 1, Default::default());
        XorNet { layer1, layer2 }
    }
}

impl Module for XorNet {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let x = xs.apply(&self.layer1).relu();
        x.apply(&self.layer2).sigmoid()
    }
}

fn main() {
    let vs = nn::VarStore::new(tch::Device::Cpu);
    let net = XorNet::new(&vs.root());

    let train_input:Vec<f32> = vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let train_output:Vec<f32> = vec![0.0, 0.0, 1.0, 1.0];

    let test_input:Vec<f32> = vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0];

    let target = Tensor::of_slice(&train_output).reshape(&[4, 1]);
    let input = Tensor::of_slice(&train_input)
        .reshape(&[4, 2])
        .to_kind(tch::Kind::Float);

    let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();

    println!("Treinando...");
    for _ in 0..10000 {
        let loss = net
            .forward(&input)
            .mse_loss(&target, tch::Reduction::Mean);
        
        opt.backward_step(&loss);
    }

    let test_input = Tensor::of_slice(&test_input)
        .reshape(&[4, 2])
        .to_kind(tch::Kind::Float);
    let output = net.forward(&test_input);
    let predicted = output.round();

    println!("Predicted XOR:");
    predicted.print();
    for i in 0..4{
        let f:f64 = predicted.double_value(&[i]);
        println!("f: {}",f);
    }
}
