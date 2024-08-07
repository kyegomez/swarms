from swarms.models.openai_function_caller import OpenAIFunctionCaller
from pydantic import BaseModel, Field


# Pydantic is a data validation library that provides data validation and parsing using Python type hints.
# It is used here to define the data structure for making API calls to retrieve weather information.
class Transaction(BaseModel):
    amount: float = Field(..., description="The amount of the transaction")
    category: str = Field(
        ...,
        description="The category of the transaction according to Xeros categories such as Dues & Subscriptions, Fees & Charges, Meals & Entertainment.",
    )
    date: str = Field(..., description="The date of the transaction")


class TransactionsToCut(BaseModel):
    transactions: list[Transaction]
    expense_analysis: str
    dollars_saved: float


# The WeatherAPI class is a Pydantic BaseModel that represents the data structure
# for making API calls to retrieve weather information. It has two attributes: city and date.


# Example usage:
# Initialize the function caller
function_caller = OpenAIFunctionCaller(
    system_prompt="You are a helpful assistant.",
    max_tokens=2000,
    temperature=0.5,
    base_model=TransactionsToCut,
)


# Task
logs = """

################################
Date	Description	Type	Amount	Balance	Action
Jul 31, 2024	MONTHLY SERVICE FEE 
Fee	-$15.00	-$986.49	
Jul 31, 2024	ORIG CO NAME:STRIPE ORIG ID:4270465600 DESC DATE: CO ENTRY DESCR:TRANSFER SEC:CCD TRACE#:111000028806182 EED:240731 IND ID:ST-R7Z6S6U2K5U1 IND NAME:KYE GOMEZ TRN: 2138806182TC	ACH credit	$18.46	-$971.49	
Jul 26, 2024	ORIG CO NAME:STRIPE ORIG ID:4270465600 DESC DATE: CO ENTRY DESCR:TRANSFER SEC:CCD TRACE#:111000027107200 EED:240726 IND ID:ST-F1B2H5X5P7A5 IND NAME:KYE GOMEZ TRN: 2087107200TC	ACH credit	$48.25	-$989.95	
Jul 24, 2024	ORIG CO NAME:STRIPE ORIG ID:4270465600 DESC DATE: CO ENTRY DESCR:TRANSFER SEC:CCD TRACE#:111000026863283 EED:240724 IND ID:ST-B3Q3I3S7G1C8 IND NAME:KYE GOMEZ TRN: 2066863283TC	ACH credit	$18.81	-$1,038.20	
Jul 23, 2024	ORIG CO NAME:STRIPE ORIG ID:4270465600 DESC DATE: CO ENTRY DESCR:TRANSFER SEC:CCD TRACE#:111000024970457 EED:240723 IND ID:ST-Y8V3O8K6B8Y2 IND NAME:KYE GOMEZ TRN: 2054970457TC	ACH credit	$48.15	-$1,057.01	
Jul 22, 2024	ORIG CO NAME:GitHub Sponsors ORIG ID:4270465600 DESC DATE: CO ENTRY DESCR:GitHub SpoSEC:CCD TRACE#:111000029548278 EED:240722 IND ID:ST-G1P1A1A3Y8L2 IND NAME:KYE GOMEZ TRN: 2049548278TC	Other	$8.33	-$1,105.16	
Jul 22, 2024	ORIG CO NAME:STRIPE ORIG ID:4270465600 DESC DATE: CO ENTRY DESCR:TRANSFER SEC:CCD TRACE#:111000029566827 EED:240722 IND ID:ST-A4F9I2H5H6I9 IND NAME:KYE GOMEZ TRN: 2049566827TC	ACH credit	$18.66	-$1,113.49	
Jul 19, 2024	ORIG CO NAME:STRIPE ORIG ID:4270465600 DESC DATE: CO ENTRY DESCR:TRANSFER SEC:CCD TRACE#:111000025982141 EED:240719 IND ID:ST-K4M7U0J6X3T3 IND NAME:KYE GOMEZ TRN: 2015982141TC	ACH credit	$19.11	-$1,132.15	
Jul 12, 2024	ORIG CO NAME:STRIPE ORIG ID:4270465600 DESC DATE: CO ENTRY DESCR:TRANSFER SEC:CCD TRACE#:111000023532836 EED:240712 IND ID:ST-L3F1Q6U7O2I4 IND NAME:KYE GOMEZ TRN: 1943532836TC	ACH credit	$1.58	-$1,151.26	
Jul 11, 2024	ORIG CO NAME:STRIPE ORIG ID:4270465600 DESC DATE: CO ENTRY DESCR:TRANSFER SEC:CCD TRACE#:111000027946637 EED:240711 IND ID:ST-T2S8O9G9L6Y6 IND NAME:KYE GOMEZ TRN: 1937946637TC	ACH credit	$19.11	-$1,152.84	
Jul 9, 2024	OVERDRAFT FEE FOR A $19.49 CARD PURCHASE - DETAILS: 0706TST* LULU'S - ALAMED WEST MENLO PA CA0############0029 07	Fee	-$34.00	-$1,171.95	
Jul 9, 2024	OVERDRAFT FEE FOR A $38.77 CARD PURCHASE - DETAILS: 0705TST* LULU'S - ALAMED WEST MENLO PA CA0############0029 07	Fee	-$34.00	-$1,137.95	
Jul 9, 2024	ORIG CO NAME:STRIPE ORIG ID:4270465600 DESC DATE: CO ENTRY DESCR:TRANSFER SEC:CCD TRACE#:111000021343065 EED:240709 IND ID:ST-G4X7Q1Z3O7P2 IND NAME:KYE GOMEZ TRN: 1911343065TC	ACH credit	$18.71	-$1,103.95	
Jul 8, 2024	OVERDRAFT FEE FOR A $66.00 ITEM - DETAILS: ORIG CO NAME:CAPITAL ONE ORIG ID:9541719318 DESC DATE:240704 CO ENTRY DESCR:CRCARDPMT SEC:CCD TRACE#:056073615999158 EED:240705 IND ID:3XS9ZC4R7RBL1JG IND NAME:KYE B GOMEZ TRN: 1875999158TC	Fee	-$34.00	-$1,122.66	
Jul 8, 2024	OVERDRAFT FEE FOR A $15.20 CARD PURCHASE - DETAILS: 0704STARBUCKS STORE 05798 MENLO PARK CA 0############0029 07	Fee	-$34.00	-$1,088.66	
Jul 8, 2024	OVERDRAFT FEE FOR A $11.35 CARD PURCHASE - DETAILS: 0703CHIPOTLE 0801 MOUNTAIN VIEW CA 0############0029 07	Fee	-$34.00	-$1,054.66	
Jul 8, 2024	OVERDRAFT FEE FOR A $26.17 CARD PURCHASE - DETAILS: 0703KFC/LJS #223 MOUNTAIN VIEW CA 0############0029 05	Fee	-$34.00	-$1,020.66	
Jul 8, 2024	TST* LULU'S - ALAMED WEST MENLO PA CA 07/06 (...0029)	Card	-$19.49	-$986.66	
Jul 8, 2024	TST* LULU'S - ALAMED WEST MENLO PA CA 07/05 (...0029)	Card	-$38.77	-$967.17	
Jul 5, 2024	OVERDRAFT FEE FOR A $13.97 CARD PURCHASE - DETAILS: 0702SAMOVAR MOUNTAIN VIEW CA 0############0029 05	Fee	-$34.00	-$928.40	
Jul 5, 2024	OVERDRAFT FEE FOR A $18.66 CARD PURCHASE - DETAILS: 0703LYFT *1 RIDE 07-01 HELP.LYFT.COM CA0############0029 01	Fee	-$34.00	-$894.40	
Jul 5, 2024	OVERDRAFT FEE FOR A $10.59 CARD PURCHASE - DETAILS: 0702PAYPAL *ELENA_SMIRNOV 402-935-7733 CA0############0029 00419	Fee	-$34.00	-$860.40	
Jul 5, 2024	ORIG CO NAME:CAPITAL ONE ORIG ID:9541719318 DESC DATE:240704 CO ENTRY DESCR:CRCARDPMT SEC:CCD TRACE#:056073615999158 EED:240705 IND ID:3XS9ZC4R7RBL1JG IND NAME:KYE B GOMEZ TRN: 1875999158TC	ACH debit	-$66.00	-$826.40	
Jul 5, 2024	UBER *TRIP SAN FRANCISCO CA 127199 07/04 (...0029)	Card	-$16.85	-$760.40	
Jul 5, 2024	STARBUCKS STORE 05798 MENLO PARK CA 07/04 (...0029)	Card	-$15.20	-$743.55	
Jul 5, 2024	CHIPOTLE 0801 MOUNTAIN VIEW CA 07/03 (...0029)	Card	-$11.35	-$728.35	
Jul 5, 2024	KFC/LJS #223 MOUNTAIN VIEW CA 07/03 (...0029)	Card	-$26.17	-$717.00	
Jul 5, 2024	ORIG CO NAME:STRIPE ORIG ID:4270465600 DESC DATE: CO ENTRY DESCR:TRANSFER SEC:CCD TRACE#:111000021739712 EED:240705 IND ID:ST-E7N6R7F0Y2B1 IND NAME:KYE GOMEZ TRN: 1871739712TC	ACH credit	$94.80	-$690.83	
Jul 3, 2024	OVERDRAFT FEE FOR A $23.68 CARD PURCHASE - DETAILS: 0701CHIPOTLE 0801 MOUNTAIN VIEW CA 0############0029 07	Fee	-$34.00	-$785.63	
Jul 3, 2024	OVERDRAFT FEE FOR A $46.59 CARD PURCHASE - DETAILS: 0702LYFT *4 RIDES 06-3 HELP.LYFT.COM CA0############0029 01	Fee	-$34.00	-$751.63	
Jul 3, 2024	SAMOVAR MOUNTAIN VIEW CA 07/02 (...0029)	Card	-$13.97	-$717.63	
Jul 3, 2024	LYFT *1 RIDE 07-01 HELP.LYFT.COM CA 07/03 (...0029)	Card	-$18.66	-$703.66	
Jul 3, 2024	PAYPAL *ELENA_SMIRNOV 402-935-7733 CA 07/02 (...0029)	Card	-$10.59	-$685.00	
Jul 2, 2024	OVERDRAFT FEE FOR A $18.35 CARD PURCHASE - DETAILS: 0629STARBUCKS STORE 05798 MENLO PARK CA 0############0029 07	Fee	-$34.00	-$674.41	
Jul 2, 2024	OVERDRAFT FEE FOR A $20.00 RECURRING CARD PURCHASE - DETAILS: 0629OPENAI *CHATGPT SUBS HTTPSOPENAI.C CA0############0029 01699	Fee	-$34.00	-$640.41	
Jul 2, 2024	OVERDRAFT FEE FOR A $31.27 CARD PURCHASE - DETAILS: 0629LULU'S ON THE ALAMEDA MENLO PARK CA 0############0029 07	Fee	-$34.00	-$606.41	
Jul 2, 2024	OVERDRAFT FEE FOR A $11.99 CARD PURCHASE - DETAILS: 0629LYFT *1 RIDE 06-27 HELP.LYFT.COM CA0############0029 01	Fee	-$34.00	-$572.41	
Jul 2, 2024	OVERDRAFT FEE FOR A $21.73 CARD PURCHASE - DETAILS: 0628SQ *BRIOCHE BAKERY & San Francisco CA0############0029 07	Fee	-$34.00	-$538.41	
Jul 2, 2024	OVERDRAFT FEE FOR A $16.04 CARD PURCHASE - DETAILS: 0628CHIPOTLE 0801 MOUNTAIN VIEW CA 0############0029 07	Fee	-$34.00	-$504.41	
Jul 2, 2024	CHIPOTLE 0801 MOUNTAIN VIEW CA 07/01 (...0029)	Card	-$23.68	-$470.41	
Jul 2, 2024	LYFT *4 RIDES 06-3 HELP.LYFT.COM CA 07/02 (...0029)	Card	-$46.59	-$446.73	
Jul 1, 2024	TACO BELL #28833 PALO ALTO CA 06/30 (...0029)	Card	-$21.80	-$400.14	
Jul 1, 2024	UBER *TRIP SAN FRANCISCO CA 336624 06/30 (...0029)	Card	-$8.16	-$378.34	
Jul 1, 2024	SAMOVAR MOUNTAIN VIEW CA 06/30 (...0029)	Card	-$15.27	-$370.18	
Jul 1, 2024	TST* DUTCH GOOSE Menlo Park CA 06/30 (...0029)	Card	-$40.23	-$354.91	
Jul 1, 2024	KEPLERS BOOKS MENLO PARK CA 06/30 (...0029)	Card	-$19.14	-$314.68	
Jul 1, 2024	LYFT *1 RIDE 06-29 HELP.LYFT.COM CA 07/01 (...0029)	Card	-$8.76	-$295.54	
Jul 1, 2024	WALGREENS #7087 MENLO PARK CA 06/29 (...0029)	Card	-$8.99	-$286.78	
Jul 1, 2024	STARBUCKS STORE 05798 MENLO PARK CA 06/29 (...0029)	Card	-$18.35	-$277.79	
Jul 1, 2024	OPENAI *CHATGPT SUBS HTTPSOPENAI.C CA 06/29 (...0029)	Card	-$20.00	-$259.44	
Jul 1, 2024	LULU'S ON THE ALAMEDA MENLO PARK CA 06/29 (...0029)	Card	-$31.27	-$239.44	
Jul 1, 2024	LYFT *1 RIDE 06-27 HELP.LYFT.COM CA 06/29 (...0029)	Card	-$11.99	-$208.17	
Jul 1, 2024	SQ *BRIOCHE BAKERY & San Francisco CA 06/28 (...0029)	Card	-$21.73	-$196.18	
Jul 1, 2024	CHIPOTLE 0801 MOUNTAIN VIEW CA 06/28 (...0029)	Card	-$16.04	-$174.45	
Jul 1, 2024	LYFT *4 RIDES 06-2 HELP.LYFT.COM CA 06/30 (...0029)	Card	-$167.26	-$158.41	
Jul 1, 2024	ORIG CO NAME:STRIPE ORIG ID:4270465600 DESC DATE: CO ENTRY DESCR:TRANSFER SEC:CCD TRACE#:111000028483776 EED:240701 IND ID:ST-D0P1O6R3S4R7 IND NAME:KYE GOMEZ TRN: 1838483776TC	ACH credit	$18.71	$8.85	
Jun 28, 2024	MONTHLY SERVICE FEE 
Fee	-$15.00	-$9.86	
Jun 28, 2024	ORIG CO NAME:STRIPE ORIG ID:1800948598 DESC DATE: CO ENTRY DESCR:TRANSFER SEC:CCD TRACE#:091000012519287 EED:240628 IND ID:ST-N8K5T9C8E2Y8 IND NAME:KYE GOMEZ TRN: 1802519287TC	ACH debit	-$175.20	$5.14	
Jun 28, 2024	LYFT *1 RIDE 06-26 HELP.LYFT.COM CA 06/28 (...0029)	Card	-$51.73	$180.34	
Jun 28, 2024	SQ *SHACK15 San Francisco CA 06/27 (...0029)	Card	-$5.37	$232.07	
Jun 28, 2024	CHIPOTLE 0801 MOUNTAIN VIEW CA 06/27 (...0029)	Card	-$25.86	$237.44	
Jun 28, 2024	PAYPAL *CANVAPTYLIM 35314369001 06/27 (...0029)	Card	-$250.00	$263.30	
Jun 27, 2024	UBER *TRIP SAN FRANCISCO CA 407732 06/26 (...0029)	Card	-$18.73	$513.30	
Jun 27, 2024	CHIPOTLE 0801 MOUNTAIN VIEW CA 06/26 (...0029)	Card	-$26.35	$532.03	
Jun 27, 2024	LULU'S ON THE ALAMEDA MENLO PARK CA 06/26 (...0029)	Card	-$30.28	$558.38	
Jun 27, 2024	LYFT *3 RIDES 06-2 HELP.LYFT.COM CA 06/27 (...0029)	Card	-$40.48	$588.66	
Jun 26, 2024	LULU'S ON THE ALAMEDA MENLO PARK CA 06/25 (...0029)	Card	-$41.21	$629.14	
Jun 26, 2024	LYFT *6 RIDES 06-2 HELP.LYFT.COM CA 06/26 (...0029)	Card	-$205.60	$670.35	
Jun 26, 2024	ORIG CO NAME:STRIPE ORIG ID:4270465600 DESC DATE: CO ENTRY DESCR:TRANSFER SEC:CCD TRACE#:111000022601747 EED:240626 IND ID:ST-M4C8I3J4I2U8 IND NAME:KYE GOMEZ TRN: 1782601747TC	ACH credit	$48.25	$875.95	
Jun 25, 2024	MCDONALDS F6641 SAN CARLOS CA 06/24 (...0029)	Card	-$16.26	$827.70	
Jun 25, 2024	SQ *SAPPORO ROCK-N-ROLL San Mateo CA 06/25 (...0029)	Card	-$52.24	$843.96	
Jun 25, 2024	LULU'S ON THE ALAMEDA MENLO PARK CA 06/24 (...0029)	Card	-$22.28	$896.20	
Jun 25, 2024	KEPLERS BOOKS MENLO PARK CA 06/24 (...0029)	Card	-$77.95	$918.48	
Jun 25, 2024	LYFT *1 RIDE 06-23 HELP.LYFT.COM CA 06/25 (...0029)	Card	-$7.99	$996.43	
Jun 25, 2024	ORIG CO NAME:STRIPE ORIG ID:4270465600 DESC DATE: CO ENTRY DESCR:TRANSFER SEC:CCD TRACE#:111000021325399 EED:240625 IND ID:ST-O1M2Y8X8B1Z1 IND NAME:KYE GOMEZ TRN: 1771325399TC	ACH credit	$9.26	$1,004.42	
Jun 24, 2024	LYFT *1 RIDE 06-22 HELP.LYFT.COM CA 06/24 (...0029)	Card	-$28.97	$995.16	
Jun 24, 2024	PY *CUN PALO ALTO PALO ALTO CA 06/23 (...0029)	Card	-$21.51	$1,024.13	
Jun 24, 2024	WALGREENS STORE 643 SA MENLO PARK CA 06/23 Purchase $5.79 Cash Back $20.00 (...0029)	Card	-$25.79	$1,045.64	
Jun 24, 2024	PAYPAL *ELENA_SMIRNOV 402-935-7733 CA 06/24 (...0029)	Card	-$10.59	$1,071.43	
Jun 24, 2024	LYFT *6 RIDES 06-2 HELP.LYFT.COM CA 06/23 (...0029)	Card	-$83.58	$1,082.02	
Jun 24, 2024	LULU'S ON THE ALAMEDA MENLO PARK CA 06/22 (...0029)	Card	-$26.35	$1,165.60	
Jun 24, 2024	LYFT *3 RIDES 06-2 HELP.LYFT.COM CA 06/22 (...0029)	Card	-$38.41	$1,191.95	
Jun 24, 2024	ORIG CO NAME:STRIPE ORIG ID:4270465600 DESC DATE: CO ENTRY DESCR:TRANSFER SEC:CCD TRACE#:111000026019819 EED:240624 IND ID:ST-M3H3N3G9F3G9 IND NAME:KYE GOMEZ TRN: 1766019819TC
"""

# Run
response = function_caller.run(
    f"Cut out all of the expenses on the transactions in the logs above that are not necessary expenses such as Meals and Entertainment and transportation, the startup is a bit tight on cash: {logs}, Analyze the expenses and provide a summary of the expenses that can be cut out and the amount of money that can be saved."
)

# The run() method of the OpenAIFunctionCaller class is used to make a function call to the API.
# It takes a string parameter that represents the user's request or query.
print(response)
