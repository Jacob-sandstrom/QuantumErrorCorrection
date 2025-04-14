# %%
import decoder as decoder

file = "d3_t3_torino"

d = decoder.Decoder(script_name=file)


# d.train("data/d3_t3_torino")


# d.test("data/d3_t3_torino_testing", "temp_dir/d3_d_t_3_250414_183608_test_model.pt")
# d.test("dist5_time3_test_data")

d.run(f"data/{file}")