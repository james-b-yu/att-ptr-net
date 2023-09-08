
	def __init__(self, disconly=False):
		""":param disconly: if True, only collect discontinuous bracketings."""
		self.disconly = disconly
		self.maxlenseen, self.sentcount = Decimal(0), Decimal(0)
		self.exact = Decimal(0)
		self.dicenoms, self.dicedenoms = Decimal(0), Decimal(0)
		self.goldb, self.candb = Counter(), Counter()  # all brackets
		self.goldfun, self.candfun = Counter(), Counter()
		self.lascores = []
		self.golddep, self.canddep = [], []
		self.goldpos, self.candpos = [], []
		# extra accounting for breakdowns:
		self.goldbcat = defaultdict(Counter)  # brackets per category
		self.candbcat = defaultdict(Counter)
		self.goldbfunc = defaultdict(Counter)  # brackets by function tag
		self.candbfunc = defaultdict(Counter)
		self.goldbatt, self.candbatt = set(), set()  # attachments per category
		self.goldrule, self.candrule = Counter(), Counter()

	def add(self, pair):
		"""Add scores from given TreePairResult object."""
		if not self.disconly or pair.cbrack or pair.gbrack:
			self.sentcount += 1
		if self.maxlenseen < pair.lengpos:
			self.maxlenseen = pair.lengpos
		self.candb.update((pair.n, a) for a in pair.cbrack.elements())
		self.goldb.update((pair.n, a) for a in pair.gbrack.elements())
		if pair.cbrack == pair.gbrack:
			if not self.disconly or pair.cbrack or pair.gbrack:
				self.exact += 1
		self.goldpos.extend(pair.gpos)
		self.candpos.extend(pair.cpos)
		self.goldfun.update((pair.n, a) for a in pair.goldfun.elements())
		self.candfun.update((pair.n, a) for a in pair.candfun.elements())
		if pair.lascore is not None:
			self.lascores.append(pair.lascore)
		if pair.ted is not None:
			self.dicenoms += pair.ted
			self.dicedenoms += pair.denom
		if pair.gdep is not None:
			self.golddep.extend(pair.gdep)
			self.canddep.extend(pair.cdep)
		# extra bookkeeping for breakdowns
		for a, n in pair.gbrack.items():
			self.goldbcat[a[0]][(pair.n, a)] += n
		for a, n in pair.cbrack.items():
			self.candbcat[a[0]][(pair.n, a)] += n
		for a, n in pair.goldfun.items():
			self.goldbfunc[a[1]][(pair.n, a)] += n
		for a, n in pair.candfun.items():
			self.candbfunc[a[1]][(pair.n, a)] += n
		for (label, indices), parent in pair.pgbrack:
			self.goldbatt.add(((pair.n, label, indices), parent))
		for (label, indices), parent in pair.pcbrack:
			self.candbatt.add(((pair.n, label, indices), parent))
		self.goldrule.update((pair.n, indices, rule)
				for indices, rule in pair.grule.elements())
		self.candrule.update((pair.n, indices, rule)
				for indices, rule in pair.crule.elements())

	def scores(self):
		"""Return a dictionary with running scores for all added sentences."""
		return dict(lr=nozerodiv(lambda: recall(self.goldb, self.candb)),
				lp=nozerodiv(lambda: precision(self.goldb, self.candb)),
				lf=nozerodiv(lambda: f_measure(self.goldb, self.candb)),
				ex=nozerodiv(lambda: self.exact / self.sentcount),
				tag=nozerodiv(lambda: accuracy(self.goldpos, self.candpos)),
				fun=nozerodiv(lambda: f_measure(self.goldfun, self.candfun)))