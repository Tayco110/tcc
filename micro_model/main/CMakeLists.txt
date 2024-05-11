idf_component_register(
    SRCS 
        "main.cc" 
        "main_functions.cc" 
        "model.cc"
        "input.cc"

    PRIV_REQUIRES spi_flash
    INCLUDE_DIRS ".")