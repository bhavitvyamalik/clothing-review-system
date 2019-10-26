from flask import Flask, render_template, request
from final_run import Reviews
from final_rating import Ratings
import operator
import logging


app = Flask(__name__)

logging.basicConfig(filename="C:/Users/bhavitvyamalik/Desktop/Training/task 1/Flask App/logname.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

logger=logging.getLogger()
logger.setLevel(logging.DEBUG)

@app.route('/')
def home():
    logger.debug("Loading Home page")
    return render_template('base.html')

@app.route('/category',methods = ['POST', 'GET'])
def function():
    if request.method=='POST':
        text = request.form['review']
        logger.debug("Text retrieved")
        A=Reviews()
        B=Ratings()
        logger.debug("Object created")
        logger.debug("Sending to function call")
        #pred_sentences = ['Wore it for my birthday. Perfect fit.']

        result=list(A.getListPrediction([text]))
        result2=list(B.getRatings([text]))

        logger.debug("Returned from function")
        var=result[0]['probabilities']
        index, value = max(enumerate(var), key=operator.itemgetter(1))

        var2=result2[0]['probabilities']
        index2, value = max(enumerate(var2), key=operator.itemgetter(1))

        index2 = "Rating: " + str(index2+1)
        #print(index,value)
        if index==0:
            msg="Category: Bottoms"
            txt="Confidence: {}".format(value)
        elif index==1:
            msg="Category: Dresses"
            txt="Confidence: {}".format(value)
        elif index==2:
            msg="Category: Intimate"
            txt="Confidence: {}".format(value)
        elif index==3:
            msg="Category: Jackets"
            txt="Confidence: {}".format(value)
        elif index==4:
            msg="Category: Tops"
            txt="Confidence: {}".format(value)
        else:
            msg="Category: Trends"
            txt="Confidence: {}".format(value)

        logger.debug("Output values decided")
        return render_template("base.html",msg=msg,txt=index2,text=text)

        logger.debug("The End")


if __name__=='__main__':
    app.run(debug=True)
