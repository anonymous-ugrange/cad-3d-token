ORIGINAL = False #False for GCO
SHAPE_LOSS = True #True for A3PL

#--------------------------original-------------------------------
if ORIGINAL:
    N_BIT=8

    END_TOKEN=["PADDING", "START", "END_SKETCH",
                    "END_FACE", "END_LOOP", "END_CURVE", "END_EXTRUSION"]

    END_PAD=7
    BOOLEAN_PAD=4

    MAX_CAD_SEQUENCE_LENGTH=272

    SKETCH_TOKEN = ["PADDING", "START", "END_SKETCH",
                    "END_FACE", "END_LOOP", "END_CURVE", "CURVE"]
    EXTRUSION_TOKEN = ["PADDING", "START", "END_EXTRUDE_SKETCH"]

    CURVE_TYPE=["Line","Arc","Circle"]

    EXTRUDE_OPERATIONS = ["NewBodyFeatureOperation", "JoinFeatureOperation",
                        "CutFeatureOperation", "IntersectFeatureOperation"]


    NORM_FACTOR=0.75
    EXTRUDE_R=1
    SKETCH_R=1

    PRECISION = 1e-5
    eps = 1e-7


    MAX_SKETCH_SEQ_LENGTH = 150
    MAX_EXTRUSION = 10
    ONE_EXT_SEQ_LENGTH = 10
    VEC_TYPE=2


    CAD_CLASS_INFO = {
        'one_hot_size': END_PAD+BOOLEAN_PAD+2**N_BIT,
        'index_size': MAX_EXTRUSION+1, # +1 for padding
        'flag_size': ONE_EXT_SEQ_LENGTH+2 # +2 for sketch and padding
    }

else:
#------------------------naive structure input---------------------
    N_BIT=8
    END_TOKEN=["PADDING", "START", "END_SKETCH", "END_FACE", "END_LOOP", "END_CURVE",
            "START_LINE", "START_ARC", "START_CIRCLE", "END_EXTRUSION"]

    END_PAD=10
    BOOLEAN_PAD=4

    MAX_CAD_SEQUENCE_LENGTH=272

    SKETCH_TOKEN = ["PADDING", "START", "END_SKETCH", "END_FACE", "END_LOOP", "END_CURVE",
                    "START_LINE", "START_ARC", "START_CIRCLE", "CURVE"]

    EXTRUSION_TOKEN = ["PADDING", "START", "END_EXTRUDE_SKETCH"]

    CURVE_TYPE=["Line","Arc","Circle"]
    OFFSET = {
        "line": 1,
        "arc": 2,
        "circle": 2,
    }

    EXTRUDE_OPERATIONS = ["NewBodyFeatureOperation", "JoinFeatureOperation",
                        "CutFeatureOperation", "IntersectFeatureOperation"]

    NORM_FACTOR=0.75
    EXTRUDE_R=1
    SKETCH_R=1

    PRECISION = 1e-5
    eps = 1e-7

    MAX_EXTRUSION = 10
    ONE_EXT_SEQ_LENGTH = 10

    CAD_CLASS_INFO = {
        'one_hot_size': END_PAD+BOOLEAN_PAD+2**N_BIT,
        'index_size': MAX_EXTRUSION+1, # +1 for padding
        'flag_size': ONE_EXT_SEQ_LENGTH+2 # +2 for sketch and padding
    }

SVG_COMMANDS = ['SOS', 'EOS', 'L', 'C']
SVG_SOS_IDX = SVG_COMMANDS.index('SOS')
SVG_EOS_IDX = SVG_COMMANDS.index('EOS')
SVG_L_IDX = SVG_COMMANDS.index('L')
SVG_C_IDX = SVG_COMMANDS.index('C')

SVG_MAX_TOTAL_LEN = 100 # maximum svg sequence length
ARGS_DIM = 256
SVG_N_ARGS = 8 # shared parameters: start=(x1, y1), end=(x2, y2)
SVG_N_COMMANDS = len(SVG_COMMANDS) 