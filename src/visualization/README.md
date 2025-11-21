### Time-series comparison: classical vs quantum-ready architecture

Below we show one swaption point (Tenor 9Y / Maturity 0.75Y):

- Blue: historical simulated prices  
- Orange: 14-day point forecast from our MLP  
- Green: one 14-day trajectory sampled from our classical GAN  
  (same conditional structure as the quantum model)

![Time series – MLP vs GAN](data/Track2_QML/plots/time_series_mlp_vs_gan_Tenor__9_Maturity__0.75.png)