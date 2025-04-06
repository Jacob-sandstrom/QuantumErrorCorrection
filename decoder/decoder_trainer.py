# %%
import decoder as decoder


d = decoder.Decoder(script_name="test")


d.train()


# d.test("dist5_time3_test_data", "temp_dir/d5_d_t_3_250406_125000_test_model.pt")
d.test("dist5_time3_test_data")
