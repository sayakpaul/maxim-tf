def test_maxim_variable_length(none_model, random_image_multiple_of_64):
    # this line will run several times with random images of different size
    # The none_model is instantiated only once, since the fixture scope is session.
    out = none_model(random_image_multiple_of_64)
