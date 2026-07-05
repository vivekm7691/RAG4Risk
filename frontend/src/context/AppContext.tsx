import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react';

type AppContextValue = {
  /** Increment after uploads/deletes so lists can refetch. */
  documentsVersion: number;
  bumpDocuments: () => void;
};

const AppContext = createContext<AppContextValue | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [documentsVersion, setDocumentsVersion] = useState(0);
  const bumpDocuments = useCallback(() => {
    setDocumentsVersion((v) => v + 1);
  }, []);

  const value = useMemo(
    () => ({ documentsVersion, bumpDocuments }),
    [documentsVersion, bumpDocuments]
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useAppContext(): AppContextValue {
  const ctx = useContext(AppContext);
  if (!ctx) {
    throw new Error('useAppContext must be used within AppProvider');
  }
  return ctx;
}
