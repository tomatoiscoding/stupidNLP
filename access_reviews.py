# accessing reviews from bestbuy
# example: page 1
import urllib2
import json
import lxml.html
from lxml.cssselect import CSSSelector
from bs4 import BeautifulSoup
import operator
import pandas as pd

# define function

class Scrape(object):
	"""docstring for Scrape"""
	def __init__(self, url):
		super(Scrape, self).__init__()
		self.url = url
	def get_item(self):
		review_res = []
		star_res = []
		response = urllib2.urlopen(self.url)
		raw = response.read()
		raw = raw.split('\n')
		raw = raw[8][14:(len(raw[8]) - 1)]
		s = json.loads(raw)
		s1 = s['BVRRSourceID']
		soup = BeautifulSoup(s1, 'html.parser')
		tree = lxml.html.fromstring(s1)
		sel = CSSSelector('#BVSubmissionPopupContainer > div.BVRRReviewDisplayStyle5BodyContent > div.BVRRReviewDisplayStyle5BodyContentPrimary > div.BVRRReviewDisplayStyle5Text > div.BVRRReviewTextContainer')
		sel1 = CSSSelector('#BVRRRatingOverall_Review_Display > div.BVRRRatingNormalOutOf')
		results = sel(tree)
		results1 = sel1(tree)
		for res in results:
			r0 = lxml.html.tostring(res)
			soup = BeautifulSoup(r0, 'html.parser')
			review = soup.get_text()
			review = review.replace('\n', '')
			review_res.append(review)
		for st in results1:
			r1 = lxml.html.tostring(st)
			soup1 = BeautifulSoup(r1, 'html.parser')
			star = soup1.get_text()
			star = star.replace('\n', '')
			star = int(star)
			star_res.append(star)
		return review_res, star_res

# get reviews and stars

res_all_review = []
res_all_star = []
for i in range(1, 311):
	url = 'http://bestbuy.ugc.bazaarvoice.com/3545w/5465700/reviews.djs?format=embeddedhtml&page=%s&scrollToTop=true' % i
	obj = Scrape(url)
	res = obj.get_item()
	res_all_review.append(res[0])
	res_all_star.append(res[1])

all_review = reduce(operator.concat, res_all_review)
all_star = reduce(operator.concat, res_all_star)
train = pd.DataFrame({
	'review': all_review,
	'star': all_star
	})
train.to_csv('/Users/bianbeilei/review.csv', sep = ',', encoding = 'utf-8')