
use cubecl::prelude::*;

#[cube(launch)]
fn double_values(
    input: &[f32],
    output: &mut [f32],
) {
    let i = ABSOLUTE_POS;

    if i < input.len() {
        output[i] = input[i] * 2.0;
    }
}

fn main() {
    println!("CubeCL GPU benchmark starting...");
}
