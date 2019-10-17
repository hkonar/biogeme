import pandas as pd
import biogeme.version as ver
import biogeme.biogeme as bio
import biogeme.database as db
import biogeme.messaging as msg
import biogeme.expressions as be

df = pd.read_csv("swissmetro.dat", '\t')
database = db.Database("swissmetro", df)

# Removing some observations can be done directly using pandas.
remove = (((database.data.PURPOSE != 1) & (database.data.PURPOSE != 3)) | (database.data.CHOICE == 0))
database.data.drop(database.data[remove].index, inplace=True)

# Parameters to be estimated
ASC_CAR = be.Beta('ASC_CAR', 0, None, None, 0)
ASC_TRAIN = be.Beta('ASC_TRAIN', 0, None, None, 0)
ASC_SM = be.Beta('ASC_SM', 0, None, None, 1)
B_TIME = be.Beta('B_TIME', 0, None, None, 0)
B_COST = be.Beta('B_COST', 0, None, None, 0)

# Definition of new variables
SM_COST = be.Variable('SM_CO') * (be.Variable('GA') == 0)
TRAIN_COST = be.Variable('TRAIN_CO') * (be.Variable('GA') == 0)

# Definition of new variables: adding columns to the database
CAR_AV_SP = be.DefineVariable('CAR_AV_SP', be.Variable('CAR_AV') * (be.Variable('SP') != 0), database)
TRAIN_AV_SP = be.DefineVariable('TRAIN_AV_SP', be.Variable('TRAIN_AV') * (be.Variable('SP') != 0), database)
TRAIN_TT_SCALED = be.DefineVariable('TRAIN_TT_SCALED', be.Variable('TRAIN_TT') / 100.0, database)
TRAIN_COST_SCALED = be.DefineVariable('TRAIN_COST_SCALED', TRAIN_COST / 100, database)
SM_TT_SCALED = be.DefineVariable('SM_TT_SCALED', be.Variable('SM_TT') / 100.0, database)
SM_COST_SCALED = be.DefineVariable('SM_COST_SCALED', SM_COST / 100, database)
CAR_TT_SCALED = be.DefineVariable('CAR_TT_SCALED', be.Variable('CAR_TT') / 100, database)
CAR_CO_SCALED = be.DefineVariable('CAR_CO_SCALED', be.Variable('CAR_CO') / 100, database)

# Definition of the utility functions
V1 = ASC_TRAIN + \
     B_TIME * TRAIN_TT_SCALED + \
     B_COST * TRAIN_COST_SCALED
V2 = ASC_SM + \
     B_TIME * SM_TT_SCALED + \
     B_COST * SM_COST_SCALED
V3 = ASC_CAR + \
     B_TIME * CAR_TT_SCALED + \
     B_COST * CAR_CO_SCALED

# Associate utility functions with the numbering of alternatives
V = {1: V1,
     2: V2,
     3: V3}

# Associate the availability conditions with the alternatives
av = {1: TRAIN_AV_SP,
      2: be.Variable('SM_AV'),
      3: CAR_AV_SP}

# Definition of the model. This is the contribution of each
# observation to the log likelihood function.
logprob = be.bioLogLogit(V, av, be.Variable('CHOICE'))

# Define level of verbosity
logger = msg.bioMessage()
logger.setSilent()
# logger.setWarning()
# logger.setGeneral()
# logger.setDetailed()

# Create the Biogeme object
biogeme = bio.BIOGEME(database, logprob)
biogeme.modelName = "01logit"

# Estimate the parameters
results = biogeme.estimate()
biogeme.createLogFile()

# Print the estimated values
betas = results.getBetaValues()
for k, v in betas.items():
    print(f"{k:10}=\t{v:.3g}")

# Get the results in a pandas table
pandasResults = results.getEstimatedParameters()
print(pandasResults)
