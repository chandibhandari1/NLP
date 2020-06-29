import requests

# check for 'MWS' examples
#json_req = {"text": "It was useless to provide many things, for we should find abundant provision in every town."}
#json_req = {"text": "I pointed to the spot where he had disappeared, and we followed the track with boats; nets were cast, but in vain."}


# check for 'EAP' examples
#json_req = {"text": "Here we barricaded ourselves, and, for the present were secure."}
#json_req = {"text": "Meantime the whole Paradise of Arnheim bursts upon the view."}



# check for 'HPL' examples
json_req = {"text": "The farm like grounds extended back very deeply up the hill, almost to Wheaton Street."}
# json_req = {"text": "Dr. Johnson, as I beheld him, was a full, pursy Man, very ill drest, and of slovenly Aspect."}

req = requests.post('http://127.0.0.1:5000/predict_author', json=json_req)
print(req.status_code)
print(req.json())